from PIL import Image

import os, struct
import numpy as np
import zlib
import imageio
import cv2
import csv
import shutil
import json

import zipfile

def unzip(zip_path, zip_type):
    assert zip_type in ["instance-filt", "label-filt"]
    target_dir = f'/tmp/{zip_type}'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    return os.path.join(target_dir, zip_type)

class RGBDFrame():
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
             return self.decompress_depth_zlib()
        else:
             raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
             return self.decompress_color_jpeg()
        else:
             raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)

class scannet_scene_reader:

    COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
    COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

    def __init__(self, root_dir, scene_name, obj_name):
        self.version = 4
        
        # Get file paths
        sens_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}.sens')
        semantic_zip_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_2d-label-filt.zip')
        instance_zip_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_2d-instance-filt.zip')

        label_file = os.path.join(root_dir, 'scannetv2-labels.combined.tsv')

        self.segm_json_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_vh_clean.aggregation.json')
        
        # Load
        self._load(sens_path)
        self.label_dir = unzip(semantic_zip_path, 'label-filt')
        self.inst_dir = unzip(instance_zip_path, 'instance-filt')

        # Find interested semantic IDX
        self.semantic_idx = None

        with open(label_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row_dict in reader:
                scannet_id = row_dict['id']
                category_name = row_dict['category']
                if category_name == obj_name:
                    self.semantic_idx = int(scannet_id)
                    break
        
        assert self.semantic_idx is not None, "provided object name not found!"
    
    def get_inst_name_map(self):
        with open(self.segm_json_path) as f:
            ann_json = json.load(f)
        return self.parse_seg_groups(ann_json)

    def _load(self, filename):
        with open(filename, 'rb') as f:
            # Read meta data
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]

            # Sensor name and intrinsics
            self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)

            # Compression types enum
            self.color_compression_type = self.COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = self.COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            
            # Frame size and numbers
            self.color_width  = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width  = struct.unpack('I', f.read(4))[0]
            self.depth_height =    struct.unpack('I', f.read(4))[0]
            self.depth_shift  =    struct.unpack('f', f.read(4))[0]
            num_frames        =    struct.unpack('Q', f.read(8))[0]
            
            # Read frames
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)
    
    def __getitem__(self, idx):
        # TODO: use dynamic image size
        image_size = (480, 640)
        assert idx >= 0
        assert idx < len(self.frames)
        depth_data = self.frames[idx].decompress_depth(self.depth_compression_type)
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
        if image_size is not None:
            depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        color = self.frames[idx].decompress_color(self.color_compression_type)
        if image_size is not None:
            color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        pose = self.frames[idx].camera_to_world
        
        # Read label
        label_path = os.path.join(self.label_dir, f"{idx}.png")
        label_map = np.array(Image.open(label_path))
        if image_size is not None:
            label_map = cv2.resize(label_map, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        label_map[label_map != self.semantic_idx] = 0
        label_map[label_map == self.semantic_idx] = 1
        label_map = label_map.astype(np.uint8)
        
        # Read instance map
        inst_path = os.path.join(self.inst_dir, f"{idx}.png")
        inst_map = np.array(Image.open(inst_path))
        if image_size is not None:
            inst_map = cv2.resize(inst_map, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        return {
            'color': color,
            'depth': depth,
            'pose': pose,
            'intrinsics_color': self.intrinsic_color,
            'semantic_label': label_map,
            'inst_label': inst_map
        }
    
    def __len__(self):
        return len(self.frames)

    @staticmethod
    def parse_seg_groups(ann_json):
        # return a dictionary
        # key: inst idx
        # value: object name
        all_inst_list = ann_json['segGroups']
        ret_obj_names = {}
        for inst in all_inst_list:
            object_idx = inst['objectId'] + 1 # from 0-indexed to 1-indexed for background
            assert object_idx > 0
            ret_obj_names[object_idx] = inst['label']
        return ret_obj_names
