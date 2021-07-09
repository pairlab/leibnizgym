#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Split an OBJ file into separate files per named object

Ignores vertex texture coordinates, polygon groups, parameter space vertices.
The individual files are named as the object they contain. The material file
(.mtl) is not split with the objects.

Run:
    $ objsplit.py /input/dir/file.obj /output/dir

Written by Balázs Dukai, https://github.com/balazsdukai
From https://gist.github.com/balazsdukai/dca936c72bd7a596fea5e4a2bb34a912
"""

import re
import os.path as p
import sys
from contextlib import contextmanager
import os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def main(file_in, dir_out):
    v_pat = re.compile(r"^v\s[\s\S]*")  # vertex
    vn_pat = re.compile(r"^vn\s[\s\S]*")  # vertex normal
    f_pat = re.compile(r"^f\s[\s\S]*")  # face
    o_pat = re.compile(r"^o\s[\s\S]*")  # named object
    ml_pat = re.compile(r"^mtllib[\s\S]*")  # .mtl file
    mu_pat = re.compile(r"^usemtl[\s\S]*")  # material to use
    s_pat = re.compile(r"^s\s[\s\S]*")  # shading
    vertices = ['None']  # because OBJ has 1-based indexing
    v_normals = ['None']  # because OBJ has 1-based indexing
    objects = {}
    faces = []
    mtllib = None
    usemtl = None
    shade = None
    o_id = None

    with open(file_in, 'r') as f_in:
        for line in f_in:
            v = v_pat.match(line)
            o = o_pat.match(line)
            f = f_pat.match(line)
            vn = vn_pat.match(line)
            ml = ml_pat.match(line)
            mu = mu_pat.match(line)
            s = s_pat.match(line)

            if v:
                vertices.append(v.group())
            elif vn:
                v_normals.append(vn.group())
            elif o:
                if o_id:
                    objects[o_id] = {'faces': faces,
                                     'usemtl': usemtl,
                                     's': shade}
                    o_id = o.group()
                    faces = []
                else:
                    o_id = o.group()
            elif f:
                faces.append(f.group())
            elif mu:
                usemtl = mu.group()
            elif s:
                shade = s.group()
            elif ml:
                mtllib = ml.group()
            else:
                # ignore vertex texture coordinates, polygon groups, parameter
                # space vertices
                pass

        if o_id:
            objects[o_id] = {'faces': faces,
                             'usemtl': usemtl,
                             's': shade}
        else:
            sys.exit("Cannot split an OBJ without named objects in it!")

    # vertex indices of a face
    fv_pat = re.compile(r"(?<= )\b[0-9]+\b", re.MULTILINE)
    # vertex normal indices of a face
    fn_pat = re.compile(r"(?<=\/)\b[0-9]+\b(?=\s)", re.MULTILINE)
    for o_id in objects.keys():
        faces = ''.join(objects[o_id]['faces'])
        f_vertices = {int(v) for v in fv_pat.findall(faces)}
        f_vnormals = {int(vn) for vn in fn_pat.findall(faces)}
        # vertex mapping to a sequence starting with 1
        v_map = {str(v): str(e) for e, v in enumerate(f_vertices, start=1)}
        vn_map = {str(vn): str(e) for e, vn in enumerate(f_vnormals, start=1)}
        faces_mapped = re.sub(fv_pat, lambda x: v_map[x.group()], faces)
        faces_mapped = re.sub(
            fn_pat, lambda x: vn_map[x.group()], faces_mapped)

        objects[o_id]['vertices'] = f_vertices
        objects[o_id]['vnormals'] = f_vnormals
        # old vertex indices are not needed anymore
        objects[o_id]['faces'] = faces_mapped

    oid_pat = re.compile(r"(?<=o\s).+")
    with suppress_stdout():
        for o_id in objects.keys():
            fname = oid_pat.search(o_id).group()
            file_out = p.join(dir_out, fname + ".obj")
            with open(file_out, 'w', newline=None) as f_out:
                if mtllib:
                    f_out.write(mtllib)

                f_out.write(o_id)

                for vertex in objects[o_id]['vertices']:
                    print(vertex)
                    f_out.write(vertices[int(vertex)])

                for normal in objects[o_id]['vnormals']:
                    f_out.write(v_normals[int(normal)])

                if objects[o_id]['usemtl']:
                    f_out.write(objects[o_id]['usemtl'])

                if objects[o_id]['s']:
                    f_out.write(objects[o_id]['s'])

                f_out.write(objects[o_id]['faces'])


if __name__ == '__main__':
    file_in = sys.argv[1]
    dir_out = sys.argv[2]
    main(file_in, dir_out)
