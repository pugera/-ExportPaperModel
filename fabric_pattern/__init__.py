bl_info = {
    'name': 'Fabric Pattern / 布の型紙', 'author': 'OpenAI', 'version': (1, 1, 0),
    'blender': (4, 2, 0), 'location': '3D View > Sidebar > Fabric Pattern',
    'description': 'Measured cloth flattening with seam/material boundaries, adaptive cuts and PNG/SVG export',
    'category': 'Mesh',
}

import bpy
import bmesh
import json
import os
import time
import numpy as np
from bpy.props import FloatProperty, IntProperty, BoolProperty, StringProperty, PointerProperty
from . import geometry as geo
from . import cloth_preview


def unwrap(work):
    bpy.ops.object.select_all(action='DESELECT')
    work.hide_set(False)
    work.select_set(True)
    bpy.context.view_layer.objects.active = work
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        bpy.ops.uv.unwrap(method='ANGLE_BASED', fill_holes=False, correct_aspect=False, margin=0.002)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    layer = work.data.uv_layers.active
    return np.array([layer.data[li].uv[:] for p in work.data.polygons for li in p.loop_indices]).reshape(-1, 3, 2)


def make_work(source, collection):
    if any(m.show_viewport for m in source.modifiers):
        raise ValueError('Apply enabled modifiers on a COPY first so seams match the final surface.')
    mesh = source.data.copy()
    mesh.name = source.name+'_FabricMesh'
    mesh.transform(source.matrix_world)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    marker = bm.edges.layers.int.get('fp_cut')
    src_layer = bm.faces.layers.int.get('fp_source_face') or bm.faces.layers.int.new('fp_source_face')
    for i, f in enumerate(bm.faces):
        f[src_layer] = i
    for e in bm.edges:
        if marker and e[marker]:
            e.seam = True
    bmesh.ops.triangulate(bm, faces=list(bm.faces), quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(mesh)
    bm.free()
    # A new UV layer preserves source UVs and ensures no inherited pins.
    mesh.uv_layers.new(name='FabricPattern')
    mesh.uv_layers.active_index = len(mesh.uv_layers)-1
    for item in mesh.uv_layers.active.data:
        item.pin_uv = False
    work = bpy.data.objects.new(source.name+'_Fabric', mesh)
    collection.objects.link(work)
    work.hide_render = True
    return work


def generate(source, settings, progress=print):
    from .export import export_patterns, create_flat_object
    if source is None or source.type != 'MESH':
        raise ValueError('Select one mesh object.')
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    started = time.time()
    selected, active = list(bpy.context.selected_objects), bpy.context.view_layer.objects.active
    collection = bpy.data.collections.new(source.name+'_FabricPattern')
    bpy.context.scene.collection.children.link(collection)
    work = None
    try:
        work = make_work(source, collection)
        mesh = work.data
        t = geo.topology(mesh)
        cuts = {e.index: 1 for e in mesh.edges if e.use_seam}
        if settings.material_seams:
            for ei, fs in enumerate(t['edge_faces']):
                if len(fs) == 2 and mesh.polygons[fs[0]].material_index != mesh.polygons[fs[1]].material_index:
                    cuts.setdefault(ei, 2)
        initial_cuts = dict(cuts)
        attempts, pending, history = {}, {}, []
        stop_reason = 'iteration_limit'
        for iteration in range(settings.max_iterations+1):
            for e in mesh.edges:
                e.use_seam = e.index in cuts
            uv = unwrap(work)
            groups, labels, adjacency = geo.components(t, cuts)
            reports = geo.measure(t, uv, groups, settings.stretch_percent/100,
                                  settings.compression_percent/100)
            failing = [r for r in reports if not r['ok']]
            history.append(dict(iteration=iteration, pieces=len(groups), failing=len(failing),
                                worst_stretch_percent=100*max(r['stretch'] for r in reports),
                                worst_compression_percent=100*max(r['compression'] for r in reports)))
            progress('Fabric pass %d: %d pieces, %d outside limits' % (iteration, len(groups), len(failing)))
            if not failing:
                stop_reason = 'within_limits'
                break
            if iteration == settings.max_iterations:
                break
            changed, projected = False, len(groups)
            for report in sorted(failing, key=lambda r: -r['error']):
                group = report['faces']
                key = int(group.min())
                old = pending.pop(key, None)
                force_split = False
                if old:
                    if report['flipped'] or report['overlaps'] or report['error'] >= old['error']*0.95:
                        for ei in old['edges']:
                            cuts.pop(ei, None)
                        force_split = True
                        changed = True
                if settings.use_darts and not force_split and not report['flipped'] and not report['overlaps'] and attempts.get(key, 0) < settings.max_darts:
                    path = geo.dart_path(t, group, cuts, mesh, report['face_error'])
                    attempts[key] = attempts.get(key, 0)+1
                    if path:
                        cuts.update({ei: 4 for ei in path})
                        pending[key] = dict(edges=path, error=report['error'])
                        changed = True
                        continue
                if projected < settings.max_pieces:
                    new_edges = geo.bisect_chart(t, group, adjacency)
                    if new_edges:
                        cuts.update({ei: 3 for ei in new_edges})
                        projected += 1
                        changed = True
            if not changed:
                stop_reason = 'piece_limit_or_unsplittable'
                break
        # Reports always correspond to the final unwrapped mesh, including on a limit stop.
        for e in mesh.edges:
            e.use_seam = e.index in cuts
        attr = mesh.attributes.get('fp_cut_kind') or mesh.attributes.new('fp_cut_kind', 'INT', 'EDGE')
        for e in mesh.edges:
            attr.data[e.index].value = cuts.get(e.index, 0)
        units = 1000*bpy.context.scene.unit_settings.scale_length if settings.use_scene_units else settings.mm_per_unit
        output = os.path.abspath(bpy.path.abspath(settings.output_dir))
        os.makedirs(output, exist_ok=True)
        # Each run has its own directory; previous patterns are never overwritten.
        from datetime import datetime
        run_dir = os.path.join(output, source.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        os.makedirs(run_dir)
        result, pieces = export_patterns(source, work, t, uv, reports, cuts, initial_cuts, settings, units, run_dir)
        result.update(history=history, stop_reason=stop_reason, elapsed_seconds=round(time.time()-started, 2),
                      blender_version=bpy.app.version_string, input_blend=bpy.data.filepath)
        with open(os.path.join(run_dir, 'report.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        flat = create_flat_object(pieces, collection, source.name+'_Pattern2D', units)
        work['fabric_report'] = os.path.join(run_dir, 'report.json')
        work['fabric_source'] = source.name
        work['fabric_status'] = result['status']
        work['fabric_mm_per_unit'] = units
        work['fabric_uv_layer'] = work.data.uv_layers.active.name
        work['fabric_stretch_limit_percent'] = settings.stretch_percent
        work['fabric_compression_limit_percent'] = settings.compression_percent
        flat['fabric_report'] = work['fabric_report']
        settings.last_report = work['fabric_report']
        work.hide_set(True)
        flat.hide_set(True)
        return result, work, flat
    except Exception:
        # Only remove the new generated collection and its newly created meshes.
        for obj in list(collection.objects):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)
        bpy.data.collections.remove(collection)
        raise
    finally:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            if obj.name in bpy.context.view_layer.objects and not obj.hide_get():
                obj.select_set(True)
        bpy.context.view_layer.objects.active = active


class FabricSettings(bpy.types.PropertyGroup):
    preview_frames: IntProperty(name='計算フレーム数',default=60,min=5,max=240)
    preview_gap_mm: FloatProperty(name='縫い目の初期隙間 (mm)',default=4,min=0,max=30)
    preview_quality: IntProperty(name='クロス計算品質',default=12,min=3,max=40)
    preview_density: FloatProperty(name='布の面密度 (g/m²)',default=300,min=10,max=3000)
    preview_stiffness: FloatProperty(name='伸縮の剛性',default=40,min=.1,max=500)
    preview_bending: FloatProperty(name='曲げ剛性',default=.15,min=.001,max=20)
    preview_sewing_force: FloatProperty(name='最大縫合力',default=1,min=.001,max=10)
    preview_time_scale: FloatProperty(name='時間倍率',default=.5,min=.05,max=1)
    preview_gravity: FloatProperty(name='重力の影響',default=0,min=0,max=1)
    preview_self_collision: BoolProperty(name='自己衝突（縫い目付近を除外）',default=False)
    stretch_percent: FloatProperty(name='許容する伸び (%)', default=5, min=0, max=100)
    compression_percent: FloatProperty(name='許容する縮み (%)', default=5, min=0, max=90)
    material_seams: BoolProperty(name='マテリアル境界を切る', default=True)
    use_darts: BoolProperty(name='自動ダーツを試す', default=True)
    max_darts: IntProperty(name='領域あたりのダーツ試行数', default=2, min=0, max=10)
    max_pieces: IntProperty(name='最大パーツ数', default=256, min=2, max=4096)
    max_iterations: IntProperty(name='最大反復回数', default=24, min=1, max=100)
    seam_allowance_mm: FloatProperty(name='縫い代 (mm)', default=10, min=0, max=100)
    use_scene_units: BoolProperty(name='シーンの単位を使う', default=True)
    mm_per_unit: FloatProperty(name='1 Blender単位の長さ (mm)', default=1000, min=0.001)
    sheet_width_mm: FloatProperty(name='配置幅 (mm)', default=1000, min=100, max=5000)
    dpi: IntProperty(name='PNG解像度 (dpi)', default=150, min=72, max=600)
    max_png_pixels: IntProperty(name='各PNGの最大辺 (px)', default=8000, min=1000, max=20000)
    output_dir: StringProperty(name='出力先', subtype='DIR_PATH', default='//fabric_output')
    last_report: StringProperty(name='最後のレポート', default='')


class FABRIC_OT_generate(bpy.types.Operator):
    bl_idname = 'fabric_pattern.generate'
    bl_label = '型紙を作成・出力'
    bl_description = '選択中のメッシュのコピーを展開し、実寸SVG・PNG・検証レポートを作成'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        try:
            result, _, _ = generate(context.active_object, context.scene.fabric_pattern)
            if result['status'] == 'PASS':
                self.report({'INFO'}, '%d pieces; all within limits. %s' % (result['piece_count'], result['output_dir']))
            else:
                self.report({'WARNING'}, 'DRAFT: %d pieces still fail. Read report.json.' % result['failed_piece_count'])
            return {'FINISHED'}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class FABRIC_OT_mark(bpy.types.Operator):
    bl_idname = 'fabric_pattern.mark_cut'
    bl_label = '選択辺を型紙の切れ目にする'
    bl_options = {'REGISTER', 'UNDO'}
    clear: BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH' and context.mode == 'EDIT_MESH'

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        layer = bm.edges.layers.int.get('fp_cut') or bm.edges.layers.int.new('fp_cut')
        for e in bm.edges:
            if e.select:
                e[layer] = 0 if self.clear else 1
                e.seam = not self.clear
        bmesh.update_edit_mesh(context.active_object.data)
        return {'FINISHED'}


class FABRIC_OT_open(bpy.types.Operator):
    bl_idname = 'fabric_pattern.open_output'
    bl_label = '出力フォルダーを開く'

    def execute(self, context):
        path = os.path.dirname(context.scene.fabric_pattern.last_report)
        if os.path.isdir(path):
            bpy.ops.wm.path_open(filepath=path)
        return {'FINISHED'}


class FABRIC_PT_panel(bpy.types.Panel):
    bl_label = '布の型紙 / Fabric Pattern'
    bl_idname = 'FABRIC_PT_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Fabric Pattern'

    def draw(self, context):
        l, s = self.layout, context.scene.fabric_pattern
        if context.scene.get('fabric_preview_scene'):
            l.label(text='下のパネルで縫製結果を確認できます')
            return
        l.label(text='手動指定: 編集モード → 辺 → シーム')
        l.label(text='布の種類: 面に別のマテリアルを割当')
        l.operator('fabric_pattern.mark_cut')
        op = l.operator('fabric_pattern.mark_cut', text='選択辺の切れ目指定を解除')
        op.clear = True
        for prop in ('material_seams', 'stretch_percent', 'compression_percent', 'use_darts'):
            l.prop(s, prop)
        box = l.box()
        box.label(text='実寸と出力')
        for prop in ('use_scene_units',):
            box.prop(s, prop)
        if not s.use_scene_units:
            box.prop(s, 'mm_per_unit')
        for prop in ('seam_allowance_mm', 'sheet_width_mm', 'dpi', 'output_dir'):
            box.prop(s, prop)
        box = l.box()
        box.label(text='自動分割の上限')
        for prop in ('max_pieces', 'max_iterations', 'max_darts', 'max_png_pixels'):
            box.prop(s, prop)
        l.operator('fabric_pattern.generate', icon='UV')
        if s.last_report:
            l.operator('fabric_pattern.open_output', icon='FILE_FOLDER')
        l.label(text='PASS = 三角形ごとの伸縮・重なり検証済')
        l.label(text='DRAFT = 未達あり。レポートを確認')


CLASSES = (FabricSettings, FABRIC_OT_generate, FABRIC_OT_mark, FABRIC_OT_open, FABRIC_PT_panel) + cloth_preview.CLASSES


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.fabric_pattern = PointerProperty(type=FabricSettings)


def unregister():
    del bpy.types.Scene.fabric_pattern
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
