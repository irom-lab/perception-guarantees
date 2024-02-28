"""Visualize the meshes from the 3D-Front dataset with trimesh.

"""

import os
import argparse
import json
import trimesh
import random

if __name__ == "__main__":
    # Process the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mesh_folder', default='/home/temp/3d-front/3D-FUTURE-model-tiny',
        nargs='?', help='path to 3D FUTURE dataset'
    )
    parser.add_argument('--seed', default=42, nargs='?', help='random seed')
    args = parser.parse_args()

    # super-category: {'Sofa': 2701, 'Chair': 1775, 'Lighting': 1921, 'Cabinet/Shelf/Desk': 5725, 'Table': 1090, 'Bed': 1124, 'Pier/Stool': 487, 'Others': 1740}
    # category_to_include = ('Chair', 'Pier/Stool')
    category_to_include = ['Chair']
    style_to_include = ['Modern', 'Industrial']
    theme_to_exclude = ['Cartoon']
    with open(os.path.join(args.mesh_folder, 'model_info.json'), 'r') as f:
        model_info = json.load(f)
    category_all = {}
    for model_ind, model in enumerate(model_info):
        super_category = model['super-category']
        # category = model['category']
        model_id = model['model_id']
        style = model['style']
        theme = model['theme']
        material = model['material']

        if super_category in category_to_include and style in style_to_include and theme not in theme_to_exclude:
            info = (model_id, style, theme, material)
            if super_category not in category_all:
                category_all[super_category] = [info]
            else:
                category_all[super_category] += [info]
    print('Visualizing furniture categories and number of models:')
    for category in category_to_include:
        print(category, len(category_all[category]))

    # Combine all the models
    model_info_all = sum(category_all.values(), [])
    flag = True

    # Randomly visualize the meshes
    raw_model_name = 'raw_model.obj'
    print("Num environments", len(model_info_all))
    for i, model_info in enumerate(model_info_all):
        # model_info = random.choice(model_info_all)
        model_id, style, theme, material = model_info
        if model_id == 'b32c48ad-02bc-450d-a858-7bddb0bb7ae9':
            flag = True
        if flag:
            piece = trimesh.load(
                os.path.join(args.mesh_folder, model_id, raw_model_name)
            )
        # if (piece.bounds[1, 0]-piece.bounds[0, 0] >1.5 or piece.bounds[1, 1]-piece.bounds[0, 1] >1.5 or piece.bounds[1, 2]-piece.bounds[0, 2] >1.5):
            print(i)
            print(f'Visualizing model_id: {model_id}')
            print(f'Style: {style}, Theme: {theme}, Material: {material}')
            print('Mesh x dimensions:', piece.bounds[:, 0])
            print('Mesh y dimensions:', piece.bounds[:, 1])
            print('Mesh z dimensions:', piece.bounds[:, 2])
            piece.show()
