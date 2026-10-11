"""Sewn cloth preview: split face corners, true sewing springs, flat rest metric.

The starting guess is pre-draped near the input surface. It is not a simulated
flat-table assembly. A separate scene keeps the original scene/cache untouched.
"""
import bpy
import json
import math
import os
from collections import defaultdict
import numpy as np
from mathutils import Vector, Quaternion
from . import geometry as geo


def find_pattern(context):
    active = context.active_object
    if active and active.get('fabric_uv_square_mm'):
        return active
    report = active.get('fabric_report') if active else None
    candidates = [o for o in context.scene.objects if o.type == 'MESH' and o.get('fabric_uv_square_mm')]
    if report:
        match = [o for o in candidates if o.get('fabric_report') == report]
        if len(match) == 1:
            return match[0]
    if active:
        match = [o for o in candidates if o.get('fabric_source') == active.name]
        if len(match) == 1:
            return match[0]
        candidates = match
    last_report = context.scene.fabric_pattern.last_report
    match = [o for o in candidates if o.get('fabric_report') == last_report]
    if len(match) == 1:
        return match[0]
    raise ValueError('型紙を先に作成し、元のオブジェクトまたは3D作業コピーを選択してください。')


def pattern_unit(work):
    value = work.get('fabric_mm_per_unit')
    if value is None:
        path = bpy.path.abspath(work.get('fabric_report', ''))
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                value = json.load(f)['settings']['mm_per_blender_unit']
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError('実寸情報を取得できません。型紙を再作成してください。')
    return float(value)/1000


def prepare_topology(work, gap_mm):
    if any(len(f.vertices) != 3 for f in work.data.polygons):
        raise ValueError('型紙用の三角形化された3D作業コピーを選択してください。')
    t = geo.topology(work.data)
    unit = pattern_unit(work)
    cuts = {e.index for e in work.data.edges if e.use_seam}
    groups, labels, _ = geo.components(t, cuts)
    layer_name = work.get('fabric_uv_layer')
    layer = work.data.uv_layers.get(layer_name) if layer_name else work.data.uv_layers.active
    if layer is None:
        raise ValueError('型紙UVがありません。型紙を再作成してください。')
    uv = np.array([layer.data[li].uv[:] for p in work.data.polygons for li in p.loop_indices]).reshape(-1,3,2)
    uv *= float(work['fabric_uv_square_mm'])/1000
    parent = list(range(len(t['faces'])*3))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    corners = [{int(v): fi*3+j for j,v in enumerate(face)} for fi,face in enumerate(t['faces'])]
    for ei,fs in enumerate(t['edge_faces']):
        if len(fs) == 2 and ei not in cuts:
            for v in work.data.edges[ei].vertices:
                a,b = root(corners[fs[0]][v]),root(corners[fs[1]][v])
                parent[b] = a
    mapping, originals, rest, piece_ids, new_faces = {},[],[],[],[]
    for fi,face in enumerate(t['faces']):
        tri=[]
        for j,v in enumerate(face):
            r = root(fi*3+j)
            if r not in mapping:
                mapping[r] = len(originals)
                originals.append(int(v))
                rest.append((float(uv[fi,j,0]),float(uv[fi,j,1]),0.))
                piece_ids.append(int(labels[fi]))
            tri.append(mapping[r])
        new_faces.append(tri)
    new_faces=np.asarray(new_faces,dtype=int)
    rest=np.asarray(rest)
    pos=t['verts'][originals]*unit
    # Loose edges are Blender's actual sewing springs, with zero target length.
    sewing=set()
    dart_pairs=0
    for ei in cuts:
        fs=t['edge_faces'][ei]
        if len(fs) != 2:
            continue
        for v in work.data.edges[ei].vertices:
            a,b=(mapping[root(corners[f][v])] for f in fs)
            if a != b:
                pair=tuple(sorted((a,b)))
                if pair not in sewing and labels[fs[0]] == labels[fs[1]]:
                    dart_pairs+=1
                sewing.add(pair)
    springs=np.asarray(sorted(sewing),dtype=int).reshape(-1,2)
    # Pull only stitched boundary vertices a small distance into their own panel.
    # This opens real gaps, including darts whose two legs share the same chart.
    face_centers=pos[new_faces].mean(axis=1)
    inward=np.zeros_like(pos)
    for corner in range(3):
        np.add.at(inward,new_faces[:,corner],face_centers-pos[new_faces[:,corner]])
    if len(springs):
        boundary=np.unique(springs)
        directions=inward[boundary]
        directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-20)
        pos[boundary]+=directions*(gap_mm/2000)
    if not np.isfinite(pos).all() or not np.isfinite(rest).all():
        raise ValueError('型紙に有限でない座標があります。')
    areas=np.linalg.norm(np.cross(rest[new_faces[:,1]]-rest[new_faces[:,0]],
                                  rest[new_faces[:,2]]-rest[new_faces[:,0]]),axis=1)/2
    if np.any(areas < 1e-14):
        raise ValueError('平面型紙に面積ゼロの三角形があります。DRAFTを修正してください。')
    return dict(positions=pos,rest=rest,faces=new_faces,springs=springs,originals=originals,
                piece_ids=piece_ids,face_pieces=labels,groups=groups,area_m2=float(areas.sum()),
                dart_springs=dart_pairs,unit=unit)


PALETTE = ((.27,.60,.66,1),(.76,.51,.30,1),(.48,.66,.43,1),(.64,.48,.69,1),
           (.79,.64,.36,1),(.37,.49,.70,1),(.69,.42,.47,1),(.48,.66,.64,1))


def create_preview(work, settings):
    data=prepare_topology(work,settings.preview_gap_mm)
    scene=bpy.data.scenes.new('Fabric Sew - '+work.get('fabric_source',work.name))
    try:
        scene.unit_settings.system='METRIC'
        scene.unit_settings.scale_length=1
        scene.render.fps=24
        scene.frame_start=1
        scene.frame_end=settings.preview_frames
        scene.sync_mode='NONE'
        scene['fabric_preview_scene']=True
        scene['fabric_origin_scene']=bpy.context.scene.name
        scene['fabric_origin_object']=work.name
        scene['fabric_preview_status']='READY'
        scene['fabric_initialization']='Pre-draped near source; flat UV metric is the cloth rest shape.'
        limits={'stretch_percent':settings.stretch_percent,'compression_percent':settings.compression_percent}
        report_path=bpy.path.abspath(work.get('fabric_report',''))
        if os.path.isfile(report_path):
            with open(report_path,encoding='utf-8') as f:
                source_report=json.load(f)
            limits.update({k:source_report['settings'][k] for k in limits})
        limits['stretch_percent']=float(work.get('fabric_stretch_limit_percent',limits['stretch_percent']))
        limits['compression_percent']=float(work.get('fabric_compression_limit_percent',limits['compression_percent']))
        scene['fabric_pattern_limits']=json.dumps(limits)
        collection=bpy.data.collections.new('Sewn cloth preview')
        scene.collection.children.link(collection)
        mesh=bpy.data.meshes.new(work.name+'_SewMesh')
        mesh.from_pydata(data['positions'].tolist(),data['springs'].tolist(),data['faces'].tolist())
        mesh.update()
        obj=bpy.data.objects.new(work.name+'_SewnCloth',mesh)
        collection.objects.link(obj)
        obj['fabric_cloth_solver']=True
        obj['fabric_pattern_object']=work.name
        obj['fabric_mm_per_unit']=1000.
        obj['fabric_report']=work.get('fabric_report','')
        obj['fabric_initial_gap_mm']=settings.preview_gap_mm
        obj['fabric_piece_count']=len(data['groups'])
        obj['fabric_dart_springs']=data['dart_springs']
        obj['fabric_sewing_pairs']=len(data['springs'])
        for name,domain,values in (('fp_original_vertex','POINT',data['originals']),
                                   ('fp_piece','FACE',data['face_pieces'])):
            attr=mesh.attributes.new(name,'INT',domain)
            attr.data.foreach_set('value',np.asarray(values,dtype=np.int32))
        spring_attr=mesh.attributes.new('fp_sewing_edge','INT','EDGE')
        pairset={tuple(pair) for pair in data['springs']}
        for edge in mesh.edges:
            spring_attr.data[edge.index].value=int(tuple(sorted(edge.vertices)) in pairset)
        for i,color in enumerate(PALETTE):
            material=bpy.data.materials.new('Fabric preview %02d' % (i+1))
            material.diffuse_color=color
            material.use_nodes=True
            bsdf=next((n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
            if bsdf is None:
                bsdf=material.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.inputs['Base Color'].default_value=color
            bsdf.inputs['Roughness'].default_value=.8
            mesh.materials.append(material)
        for face,pi in zip(mesh.polygons,data['face_pieces']):
            face.material_index=int(pi)%len(PALETTE)
            face.use_smooth=True
        uv=mesh.uv_layers.new(name='Pattern rest (metres)')
        for f in mesh.polygons:
            for vi,li in zip(f.vertices,f.loop_indices):
                uv.data[li].uv=data['rest'][vi,:2]
        obj.shape_key_add(name='Basis',from_mix=False)
        rest_key=obj.shape_key_add(name='Flat pattern rest',from_mix=False)
        rest_key.data.foreach_set('co',data['rest'].astype(np.float32).reshape(-1))
        rest_key.value=0
        rest_key.keyframe_insert(data_path='value',frame=1)
        rest_key.value=1
        rest_key.keyframe_insert(data_path='value',frame=2)
        rest_key.value=0
        cloth=obj.modifiers.new('Fabric sewing simulation','CLOTH')
        cs=cloth.settings
        cs.rest_shape_key=rest_key
        # Blender 5.1.2 can ignore the static rest key in an all-deform stack.
        # Initialize particle positions at frame 1, then feed the *unchanging*
        # flat pattern into the dynamic-rest input from frame 2. The solver's
        # positions are still integrated physically, not driven by the key.
        # A behavioral contraction test guards against silently using 3D lengths.
        cs.use_dynamic_mesh=True
        obj['fabric_rest_mode']='Flat pattern dynamic rest from frame 2; unpinned cloth'
        cs.use_sewing_springs=True
        cs.sewing_force_max=settings.preview_sewing_force
        cs.quality=settings.preview_quality
        cs.time_scale=settings.preview_time_scale
        cs.mass=max(data['area_m2']*settings.preview_density/1000/len(mesh.vertices),1e-7)
        cs.tension_stiffness=settings.preview_stiffness
        cs.compression_stiffness=settings.preview_stiffness
        cs.shear_stiffness=settings.preview_stiffness*.4
        cs.bending_stiffness=settings.preview_bending
        cs.tension_damping=10
        cs.compression_damping=10
        cs.shear_damping=10
        cs.bending_damping=2
        cs.air_damping=5
        cs.effector_weights.gravity=settings.preview_gravity
        cs.use_pressure=False
        cloth.collision_settings.use_collision=False
        cloth.collision_settings.use_self_collision=settings.preview_self_collision
        cloth.collision_settings.self_distance_min=.001
        cloth.collision_settings.self_impulse_clamp=.5
        # Seam-adjacent faces are excluded from self collision: stitch points must
        # be allowed to coincide. Interior faces can still collide when enabled.
        if len(data['springs']):
            seam_vertices=set(data['springs'].reshape(-1).tolist())
            excluded=set()
            for f in data['faces']:
                if any(int(v) in seam_vertices for v in f):
                    excluded.update(f.tolist())
            vg=obj.vertex_groups.new(name='Seam collision exclusion')
            vg.add(sorted(excluded),1.,'REPLACE')
            cloth.collision_settings.vertex_group_self_collisions=vg.name
        cloth.point_cache.frame_start=1
        cloth.point_cache.frame_end=settings.preview_frames
        scene['fabric_cloth_object']=obj.name
        scene['fabric_preview_settings']=json.dumps(dict(frames=settings.preview_frames,gap_mm=settings.preview_gap_mm,
            quality=settings.preview_quality,density_g_m2=settings.preview_density,tension=settings.preview_stiffness,
            bending=settings.preview_bending,sewing_force=settings.preview_sewing_force,gravity=settings.preview_gravity,
            time_scale=settings.preview_time_scale,self_collision=settings.preview_self_collision))
        scene.view_layers[0].objects.active=obj
        obj.select_set(True,view_layer=scene.view_layers[0])
        return scene,obj,data
    except Exception:
        # Only clean resources created for this preview, never source data.
        for o in list(scene.objects):
            mesh=o.data
            materials=list(mesh.materials) if o.type=='MESH' else []
            bpy.data.objects.remove(o,do_unlink=True)
            if mesh.users==0:
                bpy.data.meshes.remove(mesh)
            for m in materials:
                if m and m.users==0:
                    bpy.data.materials.remove(m)
        collections=list(scene.collection.children)
        bpy.data.scenes.remove(scene)
        for coll in collections:
            if coll.users==0:
                bpy.data.collections.remove(coll)
        raise


def solver_for(scene):
    obj=scene.objects.get(scene.get('fabric_cloth_object',''))
    if not obj or not any(m.type=='CLOTH' for m in obj.modifiers):
        raise ValueError('縫製プレビューのシーンを選択してください。')
    return obj


def sewing_pairs(obj):
    attr=obj.data.attributes.get('fp_sewing_edge')
    return np.array([tuple(e.vertices) for e in obj.data.edges if attr and attr.data[e.index].value],dtype=int).reshape(-1,2)


def seam_metrics(coords,pairs):
    gaps=np.linalg.norm(coords[pairs[:,0]]-coords[pairs[:,1]],axis=1)*1000 if len(pairs) else np.zeros(1)
    return dict(mean_gap_mm=float(gaps.mean()),max_gap_mm=float(gaps.max()),p95_gap_mm=float(np.percentile(gaps,95)))


def physical_strain(obj,coords):
    faces=np.array([f.vertices[:] for f in obj.data.polygons],dtype=int)
    rest=np.array([v.co[:] for v in obj.data.shape_keys.key_blocks['Flat pattern rest'].data])
    flat=rest[faces,:2]
    basis=np.stack((flat[:,1]-flat[:,0],flat[:,2]-flat[:,0]),axis=2)
    surface=np.stack((coords[faces[:,1]]-coords[faces[:,0]],coords[faces[:,2]]-coords[faces[:,0]]),axis=2)
    values=np.linalg.svd(surface@np.linalg.inv(basis),compute_uv=False)
    return dict(max_stretch_percent=float(max(0,values.max()-1)*100),
                max_compression_percent=float(max(0,1-values.min())*100))


def evaluate_frame(scene,obj,frame):
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        scene.frame_set(frame)
        depsgraph=bpy.context.evaluated_depsgraph_get()
        evaluated=obj.evaluated_get(depsgraph)
        mesh=evaluated.to_mesh()
        try:
            coords=np.empty(len(mesh.vertices)*3,dtype=np.float32)
            mesh.vertices.foreach_get('co',coords)
            return coords.reshape(-1,3).copy()
        finally:
            evaluated.to_mesh_clear()


def clear_replay(scene):
    old=scene.objects.get(scene.get('fabric_replay_object',''))
    if old and old.get('fabric_cloth_replay'):
        mesh=old.data
        bpy.data.objects.remove(old,do_unlink=True)
        if mesh.users==0:
            bpy.data.meshes.remove(mesh)
    if 'fabric_replay_object' in scene:
        del scene['fabric_replay_object']


def reset_preview(scene):
    obj=solver_for(scene)
    clear_replay(scene)
    obj.hide_viewport=False
    obj.hide_render=False
    obj.hide_set(False,view_layer=scene.view_layers[0])
    cloth=next(m for m in obj.modifiers if m.type=='CLOTH')
    if cloth.point_cache.is_baked:
        with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0],object=obj,active_object=obj,point_cache=cloth.point_cache):
            bpy.ops.ptcache.free_bake()
    # Setting the frame before cache start discards an unbaked forward simulation.
    scene.frame_set(scene.frame_start-1)
    scene.frame_set(scene.frame_start)
    scene['fabric_preview_status']='READY'
    return obj


def create_replay(scene,obj,samples):
    """Store actually simulated frames as absolute shape keys inside the blend.

    This avoids external point-cache dependencies and keeps exact frame samples
    after reopening. The original Cloth solver remains in the preview scene.
    """
    clear_replay(scene)
    mesh=obj.data.copy()
    replay=bpy.data.objects.new(obj.name+'_Playback',mesh)
    obj.users_collection[0].objects.link(replay)
    replay.shape_key_clear()
    replay['fabric_cloth_replay']=True
    replay['fabric_solver_object']=obj.name
    replay['fabric_frame_samples']=len(samples)
    replay['fabric_capture_first_frame']=scene.frame_start
    replay['fabric_capture_last_frame']=scene.frame_start+len(samples)-1
    # All frame coordinates come from Blender's evaluated Cloth modifier.
    for index,coords in enumerate(samples):
        key=replay.shape_key_add(name='Sim frame %04d' % (scene.frame_start+index),from_mix=False)
        key.data.foreach_set('co',np.asarray(coords,dtype=np.float32).reshape(-1))
        key.interpolation='KEY_LINEAR'
    keys=mesh.shape_keys
    keys.use_relative=False
    # Absolute key frames are automatically spaced by 10 Blender key-time units.
    keys.eval_time=keys.key_blocks[0].frame
    keys.keyframe_insert(data_path='eval_time',frame=scene.frame_start)
    keys.eval_time=keys.key_blocks[-1].frame
    keys.keyframe_insert(data_path='eval_time',frame=scene.frame_start+len(samples)-1)
    # Set linear interpolation using both modern layered and legacy Actions.
    action=keys.animation_data.action
    if hasattr(action,'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for fc in bag.fcurves:
                        for point in fc.keyframe_points:
                            point.interpolation='LINEAR'
    elif hasattr(action,'fcurves'):
        for fc in action.fcurves:
            for point in fc.keyframe_points:
                point.interpolation='LINEAR'
    obj.hide_viewport=True
    obj.hide_render=True
    scene['fabric_replay_object']=replay.name
    scene.view_layers[0].objects.active=replay
    replay.select_set(True,view_layer=scene.view_layers[0])
    return replay


class SimulationCapture:
    def __init__(self,scene):
        self.scene=scene
        self.obj=reset_preview(scene)
        self.pairs=sewing_pairs(self.obj)
        self.next_frame=scene.frame_start
        self.samples=[]
        self.trace=[]
        self.extent=max(max(self.obj.dimensions),.01)
        scene['fabric_preview_status']='SIMULATING'

    def step(self):
        coords=evaluate_frame(self.scene,self.obj,self.next_frame)
        if len(coords)!=len(self.obj.data.vertices) or not np.isfinite(coords).all():
            raise ValueError('フレーム%dでクロス計算が不安定になりました。品質を上げ、縫合力・隙間・時間倍率を下げてください。' % self.next_frame)
        if np.ptp(coords,axis=0).max()>self.extent*8:
            raise ValueError('クロスが大きく発散しました。縫合力・隙間・時間倍率を下げてください。')
        self.samples.append(coords)
        self.trace.append(dict(frame=self.next_frame,**seam_metrics(coords,self.pairs)))
        self.scene['fabric_preview_progress']='%d / %d' % (self.next_frame,self.scene.frame_end)
        self.next_frame+=1
        return self.next_frame>self.scene.frame_end

    def finish(self):
        replay=create_replay(self.scene,self.obj,self.samples)
        report=dict(frames=len(self.samples),sewing_pairs=len(self.pairs),
                    initial=self.trace[0],final=self.trace[-1],trace=self.trace,
                    max_vertex_displacement_mm=float(np.linalg.norm(self.samples[-1]-self.samples[0],axis=1).max()*1000),
                    rms_vertex_displacement_mm=float(np.sqrt(np.mean(np.sum((self.samples[-1]-self.samples[0])**2,axis=1)))*1000),
                    initialization=self.scene['fabric_initialization'],settings=json.loads(self.scene['fabric_preview_settings']),
                    pattern_limits=json.loads(self.scene.get('fabric_pattern_limits','{}')),
                    final_strain=physical_strain(self.obj,self.samples[-1]),
                    notes='Sewing-line surface only. No seam-allowance or dart-fold bulk; no stuffing pressure. Stiffness values are not calibrated fabric tests.')
        self.scene['fabric_simulation_report']=json.dumps(report)
        self.scene['fabric_preview_status']='RECORDED'
        self.scene.frame_set(self.scene.frame_end)
        return replay,report


def focus_preview(context,scene,obj):
    if context.window:
        context.window.scene=scene
    bounds=np.array([v.co[:] for v in obj.data.vertices])
    center=(bounds.min(axis=0)+bounds.max(axis=0))/2
    extent=max(np.ptp(bounds,axis=0).max(),.1)
    if context.screen:
        for area in context.screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active
                space.region_3d.view_location=Vector(center)
                # Front view: Z up, camera along -Y.
                space.region_3d.view_rotation=Quaternion((1,0,0),math.pi/2)
                space.region_3d.view_distance=extent*1.5
                space.region_3d.view_perspective='ORTHO'
                space.shading.color_type='MATERIAL'
                space.overlay.show_floor=False
                space.show_region_ui=True


class FABRIC_OT_preview_create(bpy.types.Operator):
    bl_idname='fabric_pattern.preview_create'
    bl_label='縫製プレビューを作成'
    bl_description='型紙の寸法を自然長として、専用シーンにクロスと縫合スプリングを作成'
    bl_options={'REGISTER'}

    def execute(self,context):
        try:
            work=find_pattern(context)
            scene,obj,data=create_preview(work,context.scene.fabric_pattern)
            focus_preview(context,scene,obj)
            self.report({'INFO'},'%dパーツ・%d本の縫合スプリング。計算・録画または再生してください。' % (len(data['groups']),len(data['springs'])))
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


class FABRIC_OT_preview_capture(bpy.types.Operator):
    bl_idname='fabric_pattern.preview_capture'
    bl_label='最後まで計算・録画'
    bl_description='クロスを1フレームずつ計算し、実際の計算結果をblend内に保存。Escで中止'
    _timer=None
    _capture=None

    @classmethod
    def poll(cls,context):
        return context.scene.get('fabric_preview_scene') and context.scene.get('fabric_preview_status')!='SIMULATING'

    def invoke(self,context,event):
        try:
            if context.screen and context.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
            self._capture=SimulationCapture(context.scene)
            context.window_manager.progress_begin(context.scene.frame_start,context.scene.frame_end)
            self._timer=context.window_manager.event_timer_add(.02,window=context.window)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        except Exception as exc:
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}

    def modal(self,context,event):
        if event.type=='ESC' or context.scene!=self._capture.scene:
            self._capture.scene['fabric_preview_status']='CANCELLED'
            self.cleanup(context)
            self.report({'INFO'},'計算を中止しました。初期状態へ戻して再実行できます。')
            return {'CANCELLED'}
        if event.type!='TIMER':
            return {'PASS_THROUGH'}
        try:
            done=self._capture.step()
            context.window_manager.progress_update(self._capture.next_frame-1)
            for area in context.screen.areas:
                area.tag_redraw()
            if done:
                replay,report=self._capture.finish()
                self.cleanup(context)
                self.report({'INFO'},'録画完了: 縫い目の平均隙間 %.3f mm' % report['final']['mean_gap_mm'])
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        except Exception as exc:
            self._capture.scene['fabric_preview_status']='ERROR'
            self.cleanup(context)
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}

    def cleanup(self,context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer=None
        context.window_manager.progress_end()

    def cancel(self,context):
        if self._capture:
            self._capture.scene['fabric_preview_status']='CANCELLED'
        self.cleanup(context)

    def execute(self,context):
        # Script/background entry point, equivalent to the modal UI path.
        try:
            self._capture=SimulationCapture(context.scene)
            while not self._capture.step():
                pass
            self._capture.finish()
            return {'FINISHED'}
        except Exception as exc:
            context.scene['fabric_preview_status']='ERROR'
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


class FABRIC_OT_preview_reset(bpy.types.Operator):
    bl_idname='fabric_pattern.preview_reset'
    bl_label='初期状態へ戻す'
    bl_description='このプレビューの録画を削除し、クロスを初期状態へ戻す'

    @classmethod
    def poll(cls,context):
        return context.scene.get('fabric_preview_scene') and context.scene.get('fabric_preview_status')!='SIMULATING'

    def execute(self,context):
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        reset_preview(context.scene)
        return {'FINISHED'}


class FABRIC_OT_preview_return(bpy.types.Operator):
    bl_idname='fabric_pattern.preview_return'
    bl_label='元のシーンへ戻る'

    def execute(self,context):
        scene=bpy.data.scenes.get(context.scene.get('fabric_origin_scene',''))
        if scene and context.window:
            context.window.scene=scene
            return {'FINISHED'}
        self.report({'ERROR'},'元のシーンが見つかりません。上部のシーン一覧から選択してください。')
        return {'CANCELLED'}


class FABRIC_PT_preview(bpy.types.Panel):
    bl_label='縫った結果のクロスプレビュー'
    bl_idname='FABRIC_PT_preview'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_category='Fabric Pattern'

    def draw(self,context):
        l,s=self.layout,context.scene.fabric_pattern
        scene=context.scene
        if scene.get('fabric_preview_scene'):
            l.label(text='状態: '+scene.get('fabric_preview_status','READY'))
            if scene.get('fabric_preview_status')=='SIMULATING':
                l.label(text=scene.get('fabric_preview_progress',''))
                l.label(text='Escで中止')
                return
            l.operator('fabric_pattern.preview_capture',icon='REC')
            l.operator('screen.animation_play',text='再生 / 一時停止',icon='PLAY')
            l.operator('fabric_pattern.preview_reset',icon='LOOP_BACK')
            if scene.get('fabric_simulation_report') and scene.get('fabric_preview_status')=='RECORDED':
                report=json.loads(scene['fabric_simulation_report'])
                l.label(text='平均縫い目隙間: %.3f mm' % report['final']['mean_gap_mm'])
                l.label(text='最大縫い目隙間: %.3f mm' % report['final']['max_gap_mm'])
                if 'final_strain' in report:
                    strain=report['final_strain']
                    l.label(text='計算後の伸び %.2f%% / 縮み %.2f%%' % (strain['max_stretch_percent'],strain['max_compression_percent']))
                    limits=json.loads(scene.get('fabric_pattern_limits','{}'))
                    if limits and (strain['max_stretch_percent']>limits['stretch_percent'] or strain['max_compression_percent']>limits['compression_percent']):
                        box=l.box()
                        box.alert=True
                        box.label(text='クロス計算後に型紙の許容量を超過',icon='ERROR')
                        box.label(text='物性・シーム・型紙形状を見直してください')
                l.label(text='録画はblend保存で保持されます')
            l.operator('fabric_pattern.preview_return',icon='BACK')
        else:
            l.label(text='先に型紙を作成してください')
            l.label(text='3D付近から縫製 / 平面型紙を自然長に使用')
            for name in ('preview_frames','preview_gap_mm','preview_quality','preview_density','preview_stiffness',
                         'preview_bending','preview_sewing_force','preview_time_scale','preview_gravity',
                         'preview_self_collision'):
                l.prop(s,name)
            l.operator('fabric_pattern.preview_create',icon='MOD_CLOTH')
            l.label(text='縫い代・ダーツの折り代の厚みは省略')


CLASSES=(FABRIC_OT_preview_create,FABRIC_OT_preview_capture,FABRIC_OT_preview_reset,FABRIC_OT_preview_return,FABRIC_PT_preview)
