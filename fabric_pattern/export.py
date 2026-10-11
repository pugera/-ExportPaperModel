"""Millimetre SVG and dependency-free, physically scaled PNG export."""
import csv
import html
import json
import math
import os
import struct
import zlib
from collections import defaultdict
import numpy as np
import bpy


def number(x):
    return '%.4f' % float(x)


def path_lines(lines):
    return ' '.join('M%s,%s L%s,%s' % tuple(number(v) for v in np.asarray(line).reshape(-1)) for line in lines)


def path_polys(polys):
    return ' '.join('M'+' L'.join(','.join(number(v) for v in p) for p in poly)+' Z' for poly in polys)


def chains_for_edges(work, t, labels, cuts):
    buckets = defaultdict(set)
    for ei in cuts:
        fs = t['edge_faces'][ei]
        if len(fs) == 2:
            pair = tuple(sorted(int(labels[f]) for f in fs))
            buckets[(pair, cuts[ei])].add(ei)
    chains = []
    for (pair, kind), edges in sorted(buckets.items()):
        vertex_edges = defaultdict(list)
        for ei in sorted(edges):
            for v in work.data.edges[ei].vertices:
                vertex_edges[v].append(ei)
        remaining = set(edges)
        while remaining:
            candidates = sorted(v for v, es in vertex_edges.items() if len(es) != 2 and any(e in remaining for e in es))
            start = candidates[0] if candidates else min(work.data.edges[min(remaining)].vertices)
            v, ordered, vertices = start, [], [start]
            while True:
                opts = [ei for ei in vertex_edges[v] if ei in remaining]
                if not opts:
                    break
                ei = min(opts)
                remaining.remove(ei)
                ordered.append(ei)
                a, b = work.data.edges[ei].vertices
                v = b if a == v else a
                vertices.append(v)
                if len(vertex_edges[v]) != 2 or v == start:
                    break
            chains.append(dict(pair=pair, kind=kind, edges=ordered, vertices=vertices))
    for i, chain in enumerate(chains):
        chain['id'] = ('D' if chain['pair'][0] == chain['pair'][1] else 'S')+'%03d' % (i+1)
    return chains


def prepare_pieces(source, work, t, uv, reports, cuts, units):
    pieces, labels, boundary = [], np.zeros(len(t['faces']), dtype=int), defaultdict(list)
    for pi, r in enumerate(reports):
        labels[r['faces']] = pi
        coords = uv[r['faces']]*r['scale']*units
        # Keep an SVG-friendly downward Y axis; piece orientation is otherwise arbitrary.
        coords[:, :, 1] *= -1
        coords -= coords.reshape(-1, 2).min(axis=0)
        dimensions = coords.reshape(-1, 2).max(axis=0)
        materials = sorted({work.data.polygons[int(fi)].material_index for fi in r['faces']})
        names = [source.material_slots[i].name if i < len(source.material_slots) else 'Unassigned' for i in materials]
        piece = dict(id='P%03d' % (pi+1), triangles=coords, faces=r['faces'], size=dimensions,
                     boundaries=[], cuts=[], stitches=[], marks=[], texts=[], dart_polys=[], fold_lines=[],
                     material=' / '.join(names), report=r, index=pi)
        piece['fabric_code'] = '/'.join('M%03d' % (i+1) for i in materials)
        for local, fi in enumerate(r['faces']):
            for k, ei in enumerate(t['face_edges'][fi]):
                if len(t['edge_faces'][ei]) == 1 or ei in cuts:
                    half = dict(piece=pi, edge=int(ei), face=int(fi),
                                verts=(int(t['faces'][fi, k]), int(t['faces'][fi, (k+1)%3])),
                                points=coords[local, [k, (k+1)%3]].copy(), center=coords[local].mean(axis=0))
                    boundary[int(ei)].append(half)
                    piece['boundaries'].append(half)
        pieces.append(piece)
    chains = chains_for_edges(work, t, labels, cuts)
    seam_table = []
    dart_edges = set()
    for chain in chains:
        sides = [[], []]
        for step, ei in enumerate(chain['edges']):
            v = chain['vertices'][step]
            hs = sorted(boundary[ei], key=lambda h: h['piece'])
            segs = [h['points'] if h['verts'][0] == v else h['points'][::-1] for h in hs]
            if len(segs) != 2:
                continue
            if chain['pair'][0] == chain['pair'][1] and sides[0]:
                if np.linalg.norm(sides[0][-1][1]-segs[1][0]) < np.linalg.norm(sides[0][-1][1]-segs[0][0]):
                    segs.reverse()
            for side, seg in zip(sides, segs):
                side.append(seg)
        if not sides[0]:
            continue
        lengths = [sum(float(np.linalg.norm(seg[1]-seg[0])) for seg in side) for side in sides]
        is_dart = chain['pair'][0] == chain['pair'][1]
        sewn_dart = False
        if is_dart:
            p = pieces[chain['pair'][0]]
            a = np.array([sides[0][0][0]]+[s[1] for s in sides[0]])
            b = np.array([sides[1][0][0]]+[s[1] for s in sides[1]])
            gaps = np.array([np.linalg.norm(a[0]-b[0]), np.linalg.norm(a[-1]-b[-1])])
            if gaps.min() < 0.02 and gaps.max() > 0.02:
                mouth_index = 0 if gaps[0] > gaps[1] else -1
                tip_index = -1 if mouth_index == 0 else 0
                p['dart_polys'].append(np.concatenate((a, b[::-1])))
                p['cuts'].append(np.array((a[mouth_index], b[mouth_index])))
                p['fold_lines'].append(np.array(((a[mouth_index]+b[mouth_index])/2, (a[tip_index]+b[tip_index])/2)))
                dart_edges.update(chain['edges'])
                sewn_dart = True
        for side_index, (pi, side) in enumerate(zip(chain['pair'], sides)):
            p = pieces[pi]
            length = lengths[side_index]
            # Sewing match points use original 3D arclength, so opposite marks correspond.
            original_lengths = [float(np.linalg.norm(t['verts'][chain['vertices'][j+1]]-t['verts'][chain['vertices'][j]])) for j in range(len(side))]
            cumulative = np.cumsum([0.]+original_lengths)
            for fraction in (0.25, 0.5, 0.75):
                target = fraction*cumulative[-1]
                j = min(int(np.searchsorted(cumulative, target, side='right')-1), len(side)-1)
                alpha = (target-cumulative[j])/max(original_lengths[j], 1e-20)
                seg = side[j]
                at = seg[0]+alpha*(seg[1]-seg[0])
                tangent = (seg[1]-seg[0])/max(np.linalg.norm(seg[1]-seg[0]), 1e-20)
                normal = np.array((-tangent[1], tangent[0]))
                half = next(h for h in boundary[chain['edges'][j]] if h['piece'] == pi and np.linalg.norm(h['points'].mean(axis=0)-seg.mean(axis=0)) < 1e-4)
                if np.dot(normal, half['center']-at) < 0:
                    normal *= -1
                if length > 15 or fraction == 0.5:
                    p['marks'].append(np.array((at-1.2*normal, at+2.2*normal)))
                if fraction == 0.5 and length > 6:
                    text_at = at+4*normal
                    # A narrow dart has two adjacent legs; one ID is legible on both.
                    if not any(label == chain['id'] and np.linalg.norm(pos-text_at) < 15
                               for pos, label, size in p['texts']):
                        p['texts'].append((text_at, chain['id'], 2.4))
        seam_table.append(dict(id=chain['id'], piece_a=pieces[chain['pair'][0]]['id'],
                               piece_b=pieces[chain['pair'][1]]['id'], type='dart_fold' if sewn_dart else 'seam',
                               origin={1:'manual',2:'material',3:'automatic_split',4:'automatic_dart'}[chain['kind']],
                               length_a_mm=round(lengths[0], 3), length_b_mm=round(lengths[1], 3),
                               difference_mm=round(abs(lengths[0]-lengths[1]), 3), edge_count=len(chain['edges'])))
    for p in pieces:
        for half in p['boundaries']:
            p['stitches'].append(half['points'])
            if half['edge'] not in dart_edges:
                p['cuts'].append(half['points'])
    return pieces, seam_table


def pack(pieces, allowance, width):
    pad, gap = allowance+12, 12
    x, y, row_h, full_w = 12., 38., 0., width
    for p in sorted(pieces, key=lambda p: (-p['size'][1], p['id'])):
        w, h = p['size']+2*pad
        h += 12
        if x+w > width and x > 12:
            x, y, row_h = 12., y+row_h+gap, 0.
        p['offset'] = np.array((x+pad, y+pad+12))
        p['box'] = np.array((x, y, w, h))
        x += w+gap
        row_h = max(row_h, h)
        full_w = max(full_w, x)
    return full_w, y+row_h+38


def svg_piece(p, offset, allowance):
    tx, ty = offset
    out = ['<g transform="translate(%s %s)">' % (number(tx), number(ty))]
    cuts = path_lines(p['cuts'])
    if allowance > 0:
        for color, w in (('#182c37', 2*allowance+0.4), ('#e1e8ed', 2*allowance)):
            out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round"/>' % (cuts, color, number(w)))
    out.append('<path d="%s" fill="#ffffff" stroke="none"/>' % path_polys(list(p['triangles'])+p['dart_polys']))
    if allowance == 0:
        out.append('<path d="%s" fill="none" stroke="#182c37" stroke-width="0.4"/>' % cuts)
    out.append('<path d="%s" fill="none" stroke="#147879" stroke-width="0.3" stroke-dasharray="2 1"/>' % path_lines(p['stitches']))
    out.append('<path d="%s" fill="none" stroke="#cb6727" stroke-width="0.3" stroke-dasharray="3 1 0.5 1"/>' % path_lines(p['fold_lines']))
    out.append('<path d="%s" fill="none" stroke="#23445a" stroke-width="0.4"/>' % path_lines(p['marks']))
    for at, label, size in p['texts']:
        out.append('<text x="%s" y="%s" font-size="%s" fill="#23445a">%s</text>' % (number(at[0]), number(at[1]), number(size), html.escape(label)))
    r = p['report']
    label = '%s | +%.2f%% / -%.2f%% | %s' % (p['id'], r['stretch']*100, r['compression']*100, 'PASS' if r['ok'] else 'DRAFT')
    out.append('<text x="0" y="%s" font-size="3.4" font-weight="bold">%s</text>' % (number(-allowance-6), label))
    out.append('<text x="0" y="%s" font-size="2.6">%s</text>' % (number(-allowance-2), html.escape(p['fabric_code']+' '+p['material'])))
    # Put ID inside a guaranteed triangle; useful after pieces are cut out.
    center = p['triangles'].reshape(-1, 2).mean(axis=0)
    cents = p['triangles'].mean(axis=1)
    at = cents[np.argmin(np.linalg.norm(cents-center, axis=1))]
    out.append('<text x="%s" y="%s" font-size="4" text-anchor="middle">%s</text>' % (number(at[0]), number(at[1]), p['id']))
    out.append('</g>')
    return '\n'.join(out)


def write_svg(path, pieces, width, height, allowance, title, subtitle):
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%smm" height="%smm" viewBox="0 0 %s %s">' % (number(width), number(height), number(width), number(height)),
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial, sans-serif" fill="#182c37">',
           '<text x="12" y="15" font-size="6" font-weight="bold">%s</text>' % html.escape(title),
           '<text x="12" y="23" font-size="3">%s</text>' % html.escape(subtitle)]
    out.extend(svg_piece(p, offset, allowance) for p, offset in pieces)
    y = height-13
    out.extend(['<path d="M12 %s h100 m-100 -2 v4 m100 -4 v4" fill="none" stroke="#182c37" stroke-width="0.4"/>' % number(y),
                '<text x="12" y="%s" font-size="3">100 mm / print at 100%%</text>' % number(y-4), '</g></svg>'])
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


# Small bitmap font keeps the install self-contained (no Pillow or browser required).
FONT = {
'0':'01110/10001/10011/10101/11001/10001/01110','1':'00100/01100/00100/00100/00100/00100/01110',
'2':'01110/10001/00001/00010/00100/01000/11111','3':'11110/00001/00001/01110/00001/00001/11110',
'4':'00010/00110/01010/10010/11111/00010/00010','5':'11111/10000/10000/11110/00001/00001/11110',
'6':'01110/10000/10000/11110/10001/10001/01110','7':'11111/00001/00010/00100/01000/01000/01000',
'8':'01110/10001/10001/01110/10001/10001/01110','9':'01110/10001/10001/01111/00001/00001/01110',
'A':'01110/10001/10001/11111/10001/10001/10001','B':'11110/10001/10001/11110/10001/10001/11110',
'C':'01111/10000/10000/10000/10000/10000/01111','D':'11110/10001/10001/10001/10001/10001/11110',
'E':'11111/10000/10000/11110/10000/10000/11111','F':'11111/10000/10000/11110/10000/10000/10000',
'G':'01111/10000/10000/10111/10001/10001/01111','H':'10001/10001/10001/11111/10001/10001/10001',
'I':'11111/00100/00100/00100/00100/00100/11111','J':'00111/00010/00010/00010/10010/10010/01100',
'K':'10001/10010/10100/11000/10100/10010/10001','L':'10000/10000/10000/10000/10000/10000/11111',
'M':'10001/11011/10101/10101/10001/10001/10001','N':'10001/11001/10101/10011/10001/10001/10001',
'O':'01110/10001/10001/10001/10001/10001/01110','P':'11110/10001/10001/11110/10000/10000/10000',
'Q':'01110/10001/10001/10001/10101/10010/01101','R':'11110/10001/10001/11110/10100/10010/10001',
'S':'01111/10000/10000/01110/00001/00001/11110','T':'11111/00100/00100/00100/00100/00100/00100',
'U':'10001/10001/10001/10001/10001/10001/01110','V':'10001/10001/10001/10001/10001/01010/00100',
'W':'10001/10001/10001/10101/10101/10101/01010','X':'10001/10001/01010/00100/01010/10001/10001',
'Y':'10001/10001/01010/00100/00100/00100/00100','Z':'11111/00001/00010/00100/01000/10000/11111',
'-':'00000/00000/00000/11111/00000/00000/00000','+':'00000/00100/00100/11111/00100/00100/00000',
'.':'00000/00000/00000/00000/00000/00110/00110','/':'00001/00010/00010/00100/01000/01000/10000',
'%':'11001/11010/00010/00100/01000/01011/10011',':':'00000/00110/00110/00000/00110/00110/00000',
' ':'00000/00000/00000/00000/00000/00000/00000','|':'00100/00100/00100/00100/00100/00100/00100'}


class Raster:
    def __init__(self, width, height, dpi, max_pixels):
        self.scale = min(dpi/25.4, max_pixels/max(width, height))
        self.dpi = self.scale*25.4
        self.pixels = np.full((int(math.ceil(height*self.scale))+1, int(math.ceil(width*self.scale))+1, 3), 255, dtype=np.uint8)

    def line(self, seg, color, width=0.35):
        a, b = np.asarray(seg)*self.scale
        radius = max(width*self.scale/2, 0.6)
        low = np.maximum(np.floor(np.minimum(a, b)-radius-1).astype(int), 0)
        high = np.minimum(np.ceil(np.maximum(a, b)+radius+1).astype(int), self.pixels.shape[1::-1])
        if np.any(high <= low):
            return
        yy, xx = np.ogrid[low[1]:high[1], low[0]:high[0]]
        d = b-a
        u = np.clip(((xx-a[0])*d[0]+(yy-a[1])*d[1])/max(float(d@d), 1e-20), 0, 1)
        mask = (xx-a[0]-u*d[0])**2+(yy-a[1]-u*d[1])**2 <= radius**2
        self.pixels[low[1]:high[1], low[0]:high[0]][mask] = color

    def triangle(self, tri, color=(255,255,255)):
        p = np.asarray(tri)*self.scale
        low = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(p.max(axis=0)+1).astype(int), self.pixels.shape[1::-1])
        if np.any(high <= low):
            return
        yy, xx = np.ogrid[low[1]:high[1], low[0]:high[0]]
        d = np.roll(p, -1, axis=0)-p
        signs = [(xx-p[j,0])*d[j,1]-(yy-p[j,1])*d[j,0] for j in range(3)]
        mask = ((signs[0]>=-0.2)&(signs[1]>=-0.2)&(signs[2]>=-0.2)) | ((signs[0]<=0.2)&(signs[1]<=0.2)&(signs[2]<=0.2))
        self.pixels[low[1]:high[1], low[0]:high[0]][mask] = color

    def text(self, at, string, height=3, color=(24,44,55)):
        x, y = np.round(np.asarray(at)*self.scale).astype(int)
        unit = max(1, int(round(height*self.scale/7)))
        for ch in string.upper():
            glyph = FONT.get(ch, FONT[' ']).split('/')
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == '1':
                        x0, y0 = x+col*unit, y+row*unit
                        x1, y1 = min(x0+unit, self.pixels.shape[1]), min(y0+unit, self.pixels.shape[0])
                        if x0 >= 0 and y0 >= 0:
                            self.pixels[y0:y1, x0:x1] = color
            x += 6*unit

    def save(self, path):
        def chunk(kind, data):
            return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data)&0xffffffff)
        h, w = self.pixels.shape[:2]
        compressor, blocks = zlib.compressobj(6), []
        for row in self.pixels:
            blocks.append(compressor.compress(b'\0'+row.tobytes()))
        blocks.append(compressor.flush())
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR', struct.pack('!IIBBBBB', w,h,8,2,0,0,0)))
            ppm = int(round(self.scale*1000))
            f.write(chunk(b'pHYs', struct.pack('!IIB', ppm,ppm,1)))
            f.write(chunk(b'IDAT', b''.join(blocks)))
            f.write(chunk(b'IEND', b''))


def write_png(path, pieces, width, height, settings, title, overview=False):
    image = Raster(width, height, settings.dpi, 3000 if overview else settings.max_png_pixels)
    sa = settings.seam_allowance_mm
    image.text((12, 9), title, 5)
    image.text((12, 20), 'MM / SEAM ALLOWANCE %.1f MM' % sa, 2.7)
    image.text((12, 26), 'DARK: CUT / TEAL: SEW / ORANGE: FOLD', 2.4)
    for p, offset in pieces:
        for color, w in (((24,44,55), 2*sa+0.4), ((225,232,237), 2*sa)):
            if w > 0:
                for seg in p['cuts']:
                    image.line(seg+offset, color, w)
        for tri in p['triangles']:
            image.triangle(tri+offset)
        if p['dart_polys']:
            from mathutils.geometry import tessellate_polygon
            from mathutils import Vector
            for poly in p['dart_polys']:
                vectors = [Vector((x,y,0)) for x,y in poly]
                for tri in tessellate_polygon([vectors]):
                    points = [vectors[v] if isinstance(v, int) else v for v in tri]
                    image.triangle(np.array([[v.x,v.y] for v in points])+offset)
        for lines, color in ((p['stitches'], (20,120,121)), (p['fold_lines'], (203,103,39)), (p['marks'], (35,68,90))):
            for seg in lines:
                image.line(seg+offset, color)
        image.text(offset+(0,-sa-10), p['id']+' '+('PASS' if p['report']['ok'] else 'DRAFT'), 3)
        image.text(offset+(0,-sa-6), '+%.2f%% / -%.2f%%' % (100*p['report']['stretch'],100*p['report']['compression']), 2.3)
        image.text(offset+(0,-sa-2),p['fabric_code'],2.2)
        for at, label, size in p['texts']:
            image.text(at+offset, label, size)
    y = height-13
    image.line(((12,y),(112,y)), (24,44,55), 0.4)
    image.line(((12,y-2),(12,y+2)), (24,44,55), 0.4)
    image.line(((112,y-2),(112,y+2)), (24,44,55), 0.4)
    image.text((12,y-6),'100 MM',3)
    image.save(path)
    return round(image.dpi, 4)


def write_overview(path, pieces, source_name, status):
    """A readable contact sheet; each piece is independently fitted to its tile."""
    columns = min(8, max(1, int(math.ceil(math.sqrt(len(pieces)*0.7)))))
    tile_w, tile_h = 110., 125.
    rows = int(math.ceil(len(pieces)/columns))
    width, height = columns*tile_w+24, rows*tile_h+48
    image = Raster(width,height,90,3000)
    image.text((12,10),source_name+' / '+status+' / '+str(len(pieces))+' PIECES',5)
    image.text((12,22),'OVERVIEW ONLY - EACH PIECE HAS ITS OWN SCALE - PRINT INDIVIDUAL SVG OR PNG',2.6)
    for index,p in enumerate(pieces):
        x,y = 12+(index%columns)*tile_w,38+(index//columns)*tile_h
        image.text((x+3,y+3),p['id']+' +%.1f/-%.1f%%' % (100*p['report']['stretch'],100*p['report']['compression']),2.6)
        image.text((x+3,y+10),'%.1f X %.1f MM' % tuple(p['size']),2.2)
        scale = min((tile_w-16)/max(p['size'][0],1e-10),(tile_h-30)/max(p['size'][1],1e-10))
        offset = np.array((x+tile_w/2-p['size'][0]*scale/2,y+22))
        for tri in p['triangles']:
            image.triangle(tri*scale+offset,(240,247,247))
        for seg in p['stitches']:
            image.line(seg*scale+offset,(20,120,121),0.3)
        for seg in p['fold_lines']:
            image.line(seg*scale+offset,(203,103,39),0.4)
        image.line(((x,y+tile_h-3),(x+tile_w-5,y+tile_h-3)),(210,220,225),0.25)
    image.save(path)


def export_patterns(source, work, t, uv, reports, cuts, initial_cuts, settings, units, output):
    pieces, seams = prepare_pieces(source, work, t, uv, reports, cuts, units)
    width, height = pack(pieces, settings.seam_allowance_mm, settings.sheet_width_mm)
    pairs = [(p, p['offset']) for p in pieces]
    all_ok = all(r['ok'] for r in reports)
    status = 'PASS' if all_ok else 'DRAFT'
    title = '%s - %s - %d pieces' % (source.name, status, len(pieces))
    subtitle = 'mm | seam allowance %.1f mm | dark: cut; teal: sew; orange: dart fold | limit +%.2f%% / -%.2f%%' % (settings.seam_allowance_mm, settings.stretch_percent, settings.compression_percent)
    write_svg(os.path.join(output,'layout.svg'), pairs, width, height, settings.seam_allowance_mm, title, subtitle)
    write_overview(os.path.join(output,'overview.png'),pieces,source.name,status)
    piece_dir = os.path.join(output,'pieces')
    os.makedirs(piece_dir)
    piece_reports = []
    for p in pieces:
        margin = settings.seam_allowance_mm+12
        offset = np.array((margin, margin+38))
        w, h = max(float(p['size'][0]+2*margin), 135.), float(p['size'][1]+2*margin+65)
        write_svg(os.path.join(piece_dir,p['id']+'.svg'), [(p,offset)], w,h,settings.seam_allowance_mm,
                  source.name+' / '+p['id'], subtitle)
        dpi = write_png(os.path.join(piece_dir,p['id']+'.png'), [(p,offset)], w,h,settings,source.name+' / '+p['id'])
        r = p['report']
        source_attr = work.data.attributes.get('fp_source_face')
        source_faces = sorted({int(source_attr.data[int(fi)].value) for fi in p['faces']}) if source_attr else []
        piece_reports.append(dict(id=p['id'], status='PASS' if r['ok'] else 'DRAFT', material=p['material'],fabric_code=p['fabric_code'],
                                  triangles=len(p['faces']), width_mm=round(float(p['size'][0]),3), height_mm=round(float(p['size'][1]),3),
                                  max_stretch_percent=round(r['stretch']*100,6), max_compression_percent=round(r['compression']*100,6),
                                  folded_triangles=r['flipped'], triangle_overlap_count=r['overlaps'],
                                  png_dpi=dpi, png_page_mm=[w,h], source_faces=source_faces))
    with open(os.path.join(output,'seam_pairs.csv'),'w',encoding='utf-8-sig',newline='') as f:
        fields=['id','piece_a','piece_b','type','origin','length_a_mm','length_b_mm','difference_mm','edge_count']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(seams)
    with open(os.path.join(output,'piece_materials.csv'),'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['piece','fabric_code','material'])
        writer.writerows((p['id'],p['fabric_code'],p['material']) for p in pieces)
    # Packed UVs remain inspectable in Blender; store the physical extent on the object.
    extent = max(width,height)
    layer = work.data.uv_layers.active
    for p in pieces:
        for tri, fi in zip(p['triangles'], p['faces']):
            for pt, li in zip(tri+p['offset'],work.data.polygons[int(fi)].loop_indices):
                layer.data[li].uv = (float(pt[0]/extent),float(1-pt[1]/extent))
    work['fabric_uv_square_mm'] = extent
    warnings = ['Geometric, isotropic triangle strain only; fabric shear/bias, thickness, stuffing pressure, ease and sewing process are not simulated.',
                'Seam allowance is a round offset envelope. Inspect narrow concavities and dart mouths before cutting.',
                'PNG includes physical resolution. Print at actual size and verify the 100 mm calibration bar. SVG is the exact vector master.',
                'Base mesh only. Apply modifiers on a duplicate before generation.']
    if not all_ok:
        warnings.insert(0,'DRAFT: some pieces exceed limits or contain folds/overlaps. Do not treat this as a validated pattern.')
    with open(os.path.join(output,'PRINT_README.txt'),'w',encoding='utf-8') as f:
        f.write(title+'\n'+subtitle+'\n\n'+'\n'.join(warnings)+'\n\nS###: sew matching IDs. D###: sew dart legs, fold along orange line.\nGrey: seam allowance. Cut outer dark outline.\nOriginal mesh and source UV are preserved.\n')
    return dict(status=status, source_object=source.name, output_dir=output, piece_count=len(pieces),
                failed_piece_count=sum(not r['ok'] for r in reports), triangles=len(t['faces']),
                manual_seam_edges=sum(v==1 for v in initial_cuts.values()), material_seam_edges=sum(v==2 for v in initial_cuts.values()),
                automatic_split_edges=sum(v==3 for v in cuts.values()), automatic_dart_edges=sum(v==4 for v in cuts.values()),
                dart_count=sum(s['type']=='dart_fold' for s in seams), seam_pair_count=len(seams),
                max_stretch_percent=max(p['max_stretch_percent'] for p in piece_reports),
                max_compression_percent=max(p['max_compression_percent'] for p in piece_reports),
                settings=dict(stretch_percent=settings.stretch_percent, compression_percent=settings.compression_percent,
                              mm_per_blender_unit=units,seam_allowance_mm=settings.seam_allowance_mm,
                              material_seams=settings.material_seams,use_darts=settings.use_darts,max_pieces=settings.max_pieces,
                              max_iterations=settings.max_iterations,png_requested_dpi=settings.dpi),
                layout_mm=[width,height], overview_to_scale=False,pieces=piece_reports,seams=seams,warnings=warnings), pieces


def create_flat_object(pieces, collection, name, units):
    verts, faces = [], []
    for p in pieces:
        for tri in p['triangles']:
            start = len(verts)
            for x,y in (tri+p['offset'])/units:
                verts.append((float(x),float(-y),0.))
            faces.append((start,start+1,start+2))
    mesh = bpy.data.meshes.new(name+'_Mesh')
    mesh.from_pydata(verts,[],faces)
    mesh.update()
    obj = bpy.data.objects.new(name,mesh)
    obj.hide_render = True
    collection.objects.link(obj)
    return obj
