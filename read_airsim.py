import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
import re

class DualViewParser:
    def __init__(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.frames = []
        self.qualified_frames = [] 
        
        for frame_number, pair in enumerate(self.root.findall('pair')):
            total_objects = int(pair.get("matches"))
            img1_data = self._parse_view(pair.find('img1'))
            img2_data = self._parse_view(pair.find('img2'))
            
            if total_objects > 20:
                self.qualified_frames.append(frame_number)
            
            # 存储原始数据
            self.frames.append({
                "img1": img1_data,
                "img2": img2_data,
                "total_objects": total_objects 
            })
    
    def _parse_view(self, img_node):
        view_data = []
        if img_node is None:
            return view_data
        for box in img_node.findall('box'):
            box_id = box.get('id')
            
            if box_id not in getattr(self, '_id_map', {}):
                if not hasattr(self, '_id_map'):
                    self._id_map = {}
                    self._used_numbers = set()
                    self._next_id = 1
                
                match = re.search(r'\d+$', box_id)
                if match:
                    num = int(match.group())
                    if num not in self._used_numbers:
                        self._id_map[box_id] = num
                        self._used_numbers.add(num)
                    else:
                        self._id_map[box_id] = self._next_id
                        self._used_numbers.add(self._next_id)
                        self._next_id += 1
                        while self._next_id in self._used_numbers:
                            self._next_id += 1
                else:
                    self._id_map[box_id] = self._next_id
                    self._used_numbers.add(self._next_id)
                    self._next_id += 1
                    while self._next_id in self._used_numbers:
                        self._next_id += 1
            
            obj_id = self._id_map[box_id]
            
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            view_data.append({
                "id": obj_id,
                "x": (xtl + xbr) / 2,
                "y": (ytl + ybr) / 2
            })
        return view_data
    
    def get_views_by_frame(self, frame_number):
        if frame_number < 0 or frame_number >= len(self.frames):
            return [], []
        return (
            self.frames[frame_number]["img1"],
            self.frames[frame_number]["img2"]
        )
    
    # 新增功能
    def get_frames_with_min_objects(self, min_objects=20):

        return [i for i, frame in enumerate(self.frames) 
               if frame["total_objects"] > min_objects]

class FrameTimestampParser:  
    def __init__(self, xml_path):
        self.timestamps = []
        self._parse_xml(xml_path)
    
    def _parse_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        
        self.timestamps = [
            pair.get("timestamp") 
            for pair in root.findall(".//pair")
        ]
    
    def get_timestamp(self, frame_number):
        if 0 <= frame_number < len(self.timestamps):
            return self.timestamps[frame_number]
        raise ValueError(f"无效帧号 {frame_number}，总帧数为 {len(self.timestamps)}")
    
class DualViewVisualizer:
    def __init__(self, 
                 parser, 
                 left_img_dir="left",
                 right_img_dir="right",
                 img_extension=".jpg"):
        self.parser = parser
        self.left_dir = Path(left_img_dir)
        self.right_dir = Path(right_img_dir)
        self.img_ext = img_extension

    def visualize_frame(self, 
                       frame_number, 
                       matches_index,
                       matches_id, 
                       left_positions,  # {id: (x, y), ...}
                       right_positions,  # {id: (x, y), ...}
                       output_size=(1920, 1080)):
        timestamp = self.parser.get_timestamp(frame_number)
        
        left_img = self._load_image(self.left_dir, timestamp)
        right_img = self._load_image(self.right_dir, timestamp)
        
        gap_size = 100
        canvas = self._create_canvas(left_img, right_img, output_size, gap_size)
        offset_x = left_img.shape[1] + gap_size
        
        # self._draw_objects(canvas[:, :offset_x], left_positions)
        # self._draw_objects(canvas[:, offset_x:], right_positions, offset_x)

        self._draw_matches(canvas, matches_index, matches_id, left_positions, right_positions, offset_x)
        return canvas

    def _load_image(self, img_dir, timestamp):
        img_path = img_dir / f"{timestamp}{self.img_ext}"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"图片加载失败: {img_path}")
        return img

    def _create_canvas(self, left_img, right_img, output_size, gap_size=100):
        h, w = output_size[1], output_size[0]
        left_resized = cv2.resize(left_img, (w, h))
        right_resized = cv2.resize(right_img, (w, h))
        
        white_gap = np.ones((h, gap_size, 3), dtype=np.uint8) * 255  # 255 for white color
        
        canvas = np.hstack([left_resized, white_gap, right_resized])
    
        return canvas

    def _draw_objects(self, canvas, positions, x_offset=0):
        for obj_id, (x, y) in positions.items():
            abs_x = int(x) + x_offset
            abs_y = int(y)
            cv2.circle(canvas, (abs_x, abs_y), 6, (0, 0, 255), -1)

    def _draw_matches(self, canvas, matches_index, matches_id, left_pos, right_pos, offset_x):
        for i, (left_index, right_index) in enumerate(matches_index):
            src = left_pos.get(left_index)
            dst = right_pos.get(right_index)
            if src and dst: 
                x1, y1 = int(src[0]), int(src[1])
                x2, y2 = int(dst[0]) + offset_x, int(dst[1])
                line_color = (0, 255, 0) if matches_id[i][0] == matches_id[i][1] else (0, 0, 255)
                cv2.circle(canvas, (x1, y1), 6, line_color, -1)
                cv2.circle(canvas, (x2, y2), 6, line_color, -1)
                
                cv2.line(canvas, (x1, y1), (x2, y2), line_color, 2)

if __name__ == "__main__":
    parser = DualViewParser("1.xml")
    
    qualified = parser.get_frames_with_min_objects()
    print(f"符合条件帧号: {qualified}")
    