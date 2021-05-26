from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivy.uix.camera import Camera
from android.permissions import request_permissions, Permission
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
import time
import cv2
import numpy as np
import os

request_permissions([
    Permission.CAMERA,
    Permission.WRITE_EXTERNAL_STORAGE,
    Permission.READ_EXTERNAL_STORAGE
])

Builder.load_string('''
<CameraWindow>:
    orientation: 'vertical'
    Camera:
        index: 1
        id: camera
        resolution: (1920,1080)
        play: True
        allow_stretch: True
        canvas.before:
            PushMatrix
            Rotate:
                angle: -90
                origin: self.center
        canvas.after:
            PopMatrix
    Button:
        text: 'Detect'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
<ImagePreprocessing>:
    orientation: 'vertical'
    Image:
        id: img
    Button:
        text: 'Back to camera'
        size_hint_y: None
        height: '48dp'
        on_press: root.return_to_camera()
''')
    
class CameraWindow(BoxLayout):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_name = ''
        self.image_height = 0
        self.image_width = 0
        self.labels = ['correct_mask', 'unmasked', 'incorrect_mask']
        
    def capture(self):
        camera = self.ids['camera']
        path = os.getcwd()
        timestr = time.strftime('%Y%m%d_%H%M%S')
        self.image_name = f'{path}/camera_{timestr}.png'
        camera.export_to_png(self.image_name)
        self.yolov4_tiny_model()
        
    def read_image(self):
        image = cv2.imread(self.image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def image_shape(self, image):
        return image.shape[0], image.shape[1]
    
    def colors_init(self):
        colors = ['0,255,0','255,0,0','0,0,255']
        colors = [np.array(every_color.split(',')).astype('int') for every_color in colors]
        colors = np.array(colors)
        colors = np.tile(colors,(16,1))
        
        return colors
    
    def model_output_layer(self, model):
        model_layers = model.getLayerNames()
        model_output_layer = [model_layers[yolo_layer[0] - 1] for yolo_layer in model.getUnconnectedOutLayers()]
        
        return model_output_layer
    
    def bounding_box_coordinates(self, object_detection):
        bounding_box = object_detection[0:4] * np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        (x, y, box_width, box_height) = bounding_box.astype("int")
        box_x = int(x - (box_width / 2))
        box_y = int(y - (box_height / 2))
        
        return box_x, box_y, box_width, box_height
        
    def nms_update(self, predicted_class_id, prediction_confidence, object_detection, nms_classes, nms_confidences, nms_boxes):
        box_x, box_y, box_width, box_height = self.bounding_box_coordinates(object_detection)
        
        nms_classes.append(predicted_class_id)
        nms_confidences.append(float(prediction_confidence))
        nms_boxes.append([box_x, box_y, int(box_width), int(box_height)])
        
        return nms_classes, nms_confidences, nms_boxes
    
    def non_max_supression(self, model_detection_layers):
        nms_classes = []
        nms_confidences = []
        nms_boxes = []
        for object_detection_layer in model_detection_layers:
            for object_detection in object_detection_layer:
                scores = object_detection[5:]
                predicted_class_id = np.argmax(scores)
                prediction_confidence = scores[predicted_class_id]
                
                if prediction_confidence > 0.50:
                    nms_classes, nms_confidences, nms_boxes = self.nms_update(predicted_class_id, prediction_confidence, object_detection, nms_classes, nms_confidences, nms_boxes)
                    
        return nms_classes, nms_confidences, nms_boxes
    
    def get_predicted_class_id(self, best_nms_class_score, nms_classes):
        predicted_class_id = nms_classes[best_nms_class_score]
        return predicted_class_id
    
    def get_predicted_class_label(self, predicted_class_id):
        predicted_class_label = self.labels[predicted_class_id]
        return predicted_class_label
    
    def get_prediction_confidence(self, best_nms_class_score, nms_confidences):
        prediction_confidence = nms_confidences[best_nms_class_score]
        return prediction_confidence
    
    def bbox_coords(self, coord_dot, lenght):
        return coord_dot + lenght
    
    def draw_rectangle(self, image, box_x, box_y, box_x_end, box_y_end, box_color):
        cv2.rectangle(image, (box_x, box_y), (box_x_end, box_y_end), box_color, 5)
        
    def put_text(self, image, predicted_class_label, box_x, box_y, box_color):
        cv2.putText(image, predicted_class_label, (box_x, box_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    def draw_bbox(self, best_nms_score, nms_classes, nms_confidences, nms_boxes, image, colors):
        for max_valueid in best_nms_score:
            best_nms_class_score = max_valueid[0]
            box = nms_boxes[best_nms_class_score]
            box_x = box[0]
            box_y = box[1]
            box_width = box[2]
            box_height = box[3]
            
            predicted_class_id = self.get_predicted_class_id(best_nms_class_score, nms_classes)
            predicted_class_label = self.get_predicted_class_label(predicted_class_id)
            prediction_confidence = self.get_prediction_confidence(best_nms_class_score, nms_confidences)
        
            box_x_end = self.bbox_coords(box_x, box_width)
            box_y_end = self.bbox_coords(box_y, box_height)
        
            box_color = colors[predicted_class_id]
        
            box_color = [int(c) for c in box_color]
        
            predicted_class_label = f'{predicted_class_label}: {round(prediction_confidence * 100, 2)}'
            
            self.draw_rectangle(image, box_x, box_y, box_x_end, box_y_end, box_color)
            self.put_text(image, predicted_class_label, box_x, box_y, box_color)
            
    def image_to_texture(self, image):
        buf = image.tostring()
        image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return image_texture
                            
    def yolov4_tiny_model(self):
        image = self.read_image()
        
        self.image_height, self.image_width = self.image_shape(image)

        blob = cv2.dnn.blobFromImage(image, 0.003922, (416, 416), crop=False)

        colors = self.colors_init()

        model = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg','yolov4-tiny_best.weights')
        
        model_output_layer = self.model_output_layer(model)
        
        model.setInput(blob)
        model_detection_layers = model.forward(model_output_layer)
        
        nms_classes = []
        nms_boxes = []
        nms_confidences = []
        
        nms_classes, nms_confidences, nms_boxes = self.non_max_supression(model_detection_layers)     
        
        best_nms_score = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, 0.5, 0.4)

        self.draw_bbox(best_nms_score, nms_classes, nms_confidences, nms_boxes, image, colors)
        
        cv2.imwrite('test.png', image)
        image = cv2.flip(image, 0)
        
        image_texture = self.image_to_texture(image)
            
        main_app.photo_page.update_source(image_texture)
        
        main_app.screen_manager.current = 'ImagePreprocessing'
        
class ImagePreprocessing(BoxLayout):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def return_to_camera(self):
        main_app.screen_manager.current = 'CameraWindow'
    
    def update_source(self, source_img):
        self.ids.img.texture = source_img
        

class MainApp(App):

    def build(self):
        self.screen_manager = ScreenManager()
        
        self.camera_page = CameraWindow()
        screen = Screen(name='CameraWindow')
        screen.add_widget(self.camera_page)
        self.screen_manager.add_widget(screen)
        
        self.photo_page = ImagePreprocessing()
        screen = Screen(name='ImagePreprocessing')
        screen.add_widget(self.photo_page)
        self.screen_manager.add_widget(screen)
        
        return self.screen_manager

if __name__ == '__main__':
    main_app = MainApp()
    main_app.run()
