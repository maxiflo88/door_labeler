import json
import cv2
import os
import numpy as np
import PIL
from tkinter import *
from colormap import rgb2hex
import imutils
from door import *
class image:
    def __init__(self, img_path, master):
        self.img_path=img_path
        self.master=master
        self.segm_path=img_path.replace('img','segm')
        self.doors=[]
        self.labels=[{'id':1, 'name':'kreisa 1','color':'#abeb34'}, 
                    {'id':2, 'name':'kreisa 2','color':'#93eb34' },
                    {'id':3, 'name':'kreisa 3','color':'#83eb34' },
                    {'id':4, 'name':'kreisa 4','color':'#53eb34' },
                    {'id':5, 'name':'kreisa 5','color':'#34eb3a' },
                    {'id':6, 'name':'kreisa 6','color':'#34eb68' },
                    {'id':7, 'name':'laba 1','color':'#eb3434' },
                    {'id':8, 'name':'laba 2','color':'#eb4634' },
                    {'id':9, 'name':'laba 3','color':'#eb5934' },
                    {'id':10, 'name':'laba 4','color':'#eb7734' },
                    {'id':11, 'name':'laba 5','color':'#eb8f34' },
                    {'id':12, 'name':'laba 6','color':'#ebab34' }]
        self.keypoint_connection_rules=[[1, 2, '#20B2AA'],
                                        [2, 8, '#820068'],
                                        [8, 7, '#F0553A'],
                                        [7, 1, '#78825E'], 
                                        [3, 4, '#34e1eb'], 
                                        [4, 5, '#34abeb'], 
                                        [5, 6, '#eb34eb'],
                                        [8, 9, '#80d9a2'],
                                        [9, 10, '#80bdd9'],
                                        [10, 11, '#34ebc3'],
                                        [11, 12, '#4ecfbd']]
        self.ids=0
        self.detectJsonFile(self.img_path.replace('.png', '.json'))

    def detectJsonFile(self, json_path):

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                for d in data['annotations']:
                    bbox=d['bbox']
                    keypoints=d['keypoints']
                    keypoints=self.fromCocoFormatKeypoints(keypoints)
                    if len(keypoints)==4:
                        keypoints=self.oldKeypointFormat(keypoints)
                    self.addDoors(bbox, keypoints)


    def save(self):   
        #TODO kad pabeigts samainit lai iet self.img_path un lade ari sadus failus
        im = PIL.Image.open(self.segm_path)
        width, height = im.size
        save_path=self.img_path.replace('.png','.json')
        images=[]
        annotations=[]
        images.append({'height': height, 'width': width, 'file_name': os.path.basename(self.img_path)})
        for door in self.doors:
            keypoints, num_keypoints=door.toCocoFormatKeypoints(self.labels)
            annotations.append({'bbox': door.boundBox,'keypoints': keypoints, 'num_keypoints': num_keypoints})
        cocoDataset = {
                        "images": images,
                        "annotations": annotations
                        }
        with open(save_path, 'w') as jsonfile:
            json.dump(cocoDataset, jsonfile)
        print(f'Saved {save_path}')

    def fromCocoFormatKeypoints(self, keypoints):
        '''
        Read list of keypoints from Coco format [x, y, v, x, y, v] to [[x, y, 1][ x, y, 2]]
        '''
        key=[]
        range1=0
        range2=3
        for ind in range(len(keypoints)//3):
            keypoint=keypoints[range1:range2]
            if keypoint[2]==2:
                key.append([*keypoint[:2], ind+1])
            range1=range2
            range2+=3
        return key

    def oldKeypointFormat(self, keypoints):
        '''
        Old format with only 4 keypoints corrects ids
        '''
        keypoints[2][2]=7
        keypoints[3][2]=8

        return keypoints

    def addDoors(self, boundBox, keypoints=None):
        if keypoints:
            self.doors.append(door(self.ids, boundBox, keypoints))
        else:
            self.doors.append(door(self.ids, boundBox))
        
        self.ids+=1

    def resize(self, img, wwidth, wheight):
        height, width=img.shape[:2]
        newWidth=wwidth
        self.ratio=width/wwidth
        newHeight = int(height/self.ratio)
        if newHeight>wheight:
            self.ratio=height/wheight
            newHeight=wheight
            newWidth=int(width/self.ratio)
        return cv2.resize(img, (newWidth, newHeight))

    def detect_keypoints(self):
        if self.doors:
            img=cv2.imread(self.segm_path, cv2.IMREAD_GRAYSCALE)
            for door in self.doors:
                x, y, w, h=np.array(door.boundBox, dtype=np.int)
                mask=np.ones_like(img, dtype=np.uint8)*255
                mask[y:y+h,x:x+w]=img[y:y+h,x:x+w]
                door.keypoints=list(self._getKeypoints(mask))

    def _getKeypoints(self, img):

        img = np.float32(img)   
        dst = cv2.cornerHarris(img,5,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
        return corners

    def detect_doors(self):

        im_in = cv2.imread(self.segm_path, cv2.IMREAD_GRAYSCALE)
        # Threshold.
        # Set values equal to or above 220 to 0.
        # Set values below 220 to 255.
        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        #blur it slightly
        gray = cv2.GaussianBlur(~im_out, (7, 7), 0)
        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours individually
        for c in cnts:
            self.addDoors(cv2.boundingRect(c))

    def drawDoors(self, canvas):

        if self.doors:
            for door in self.doors:
                color=np.random.randint(0, 255, (3)).tolist()
                radius=5
                x, y, w, h=door.boundBox
                x1=int(x/self.ratio)
                y1=int(y/self.ratio)
                x2=int((w)/self.ratio)
                y2=int((h)/self.ratio)
                canvas.create_rectangle(x1, y1, x1+x2, y1+y2, activefill=rgb2hex(*color), stipple="gray50")
                if door.keypoints:
                    for ind, keypoint in enumerate(door.keypoints):
                        x, y, label=keypoint
                        pointx=int(x/self.ratio)
                        pointy=int(y/self.ratio)
                        label_colors=['']
                        label_colors.extend([color['color'] for color in self.labels])

                        point=canvas.create_oval(pointx-radius, pointy-radius, pointx+radius, pointy+radius, fill=label_colors[label], tags='points', activefill=rgb2hex(*color), outline="")
                        canvas.tag_bind(point,"<Button-1>", lambda event, ind=ind: self.show_options(event, [canvas, door.ids, ind]))
                self.draw_lines(canvas)

    def draw_lines(self, canvas):
        if len(canvas.find_withtag('lines')) != 0:
            for line_id in canvas.find_withtag('lines'):
                canvas.delete(line_id)
        radius=5
        label_colors={color['color']:None for color in self.labels}
        connections=[]
        #TODO sadalit pusi no f-jas  isaka f-ja
        for key_connection in self.keypoint_connection_rules:
            point1=None
            point2=None
            for label in self.labels:
                if key_connection[0]==label['id']: point1=label['color']
                elif key_connection[1]==label['id']: point2=label['color']

                if point1!=None and point2!=None: 
                    connections.append([point1, point2, key_connection[2]])
                    break
                
        point_ids=canvas.find_withtag('points')

        for point_id in point_ids:
            for key in label_colors:
                if canvas.itemcget(point_id, "fill")==key:
                    label_colors[key]=canvas.coords(point_id)
                    break

        for connection in connections:
            if label_colors[connection[0]]!=None and label_colors[connection[1]]!=None:
                x1, y1, _, _= label_colors[connection[0]]
                x2, y2, _, _= label_colors[connection[1]]
                canvas.create_line(x1+radius, y1+radius, x2+radius, y2+radius, tags='lines', width=3, fill=connection[2])
        

    def show_options(self, *args):
        event=args[0]
        pressed_id = event.widget.find_closest(event.x, event.y)[0]
        selected_key=args[1]
        rcmenu = Menu(self.master, tearoff=0)
        for label in self.labels:
            #generate menu list
            label_id=label['id']
            rcmenu.add_command(label=label['name'], command = lambda label_id=label_id:self.selected_menu_options(label_id, pressed_id, selected_key))
        rcmenu.post(event.x_root, event.y_root)

    def selected_menu_options(self, *event):
        label_colors=['']
        label_colors.extend([color['color'] for color in self.labels])
        label=event[0]
        pressed_id=event[1]
        canvas=event[2][0]
        doors_id=event[2][1]
        keypoint_id=event[2][2]
        point_ids=canvas.find_withtag('points')
        for point_id in point_ids:
            if canvas.itemcget(point_id, "fill")==label_colors[label]:
                canvas.itemconfig(point_id, fill='')

        canvas.itemconfig(pressed_id, fill=label_colors[label])
        self.doors[doors_id].addKeypointLabel(keypoint_id, label) 
        self.draw_lines(canvas)

    def draw(self, canvas, wwidth, wheight):
        
        canvas.delete("all")
        img = cv2.imread(self.segm_path)
        resized=self.resize(img, wwidth, wheight)
        resized=PIL.Image.fromarray(resized)       
        canvas.image=PIL.ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')
        self.drawDoors(canvas)
