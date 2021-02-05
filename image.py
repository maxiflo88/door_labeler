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
        self.ids=0
        self.detectJsonFile(self.segm_path.replace('.png', '.json'), self.img_path.replace('.png', '.json'))

    def detectJsonFile(self, json_path1, json_path2):

        if os.path.exists(json_path2):
            with open(json_path2, 'r') as f:
                data = json.load(f)
                bbox=data["annotations"][0]['bbox']
                keypoints=data["annotations"][0]['keypoints']
                self.addDoors(bbox, keypoints)

        elif os.path.exists(json_path1):
            with open(json_path1, 'r') as f:
                data = json.load(f)
                bbox=data["annotations"][0]['bbox']
                self.addDoors(bbox)

        

    def addDoors(self, boundBox, keypoints=None):
        if keypoints:
            key=[]
            if keypoints[2]==2:
                key.append([*keypoints[:2], 1])
            if keypoints[5]==2:
                key.append([*keypoints[3:5], 2])
            if keypoints[8]==2:
                key.append([*keypoints[6:8], 3])
            if keypoints[11]==2:
                key.append([*keypoints[9:11], 4])
            self.doors.append(door(self.ids, boundBox, key))
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
                thickness=3
                # cv2.rectangle(img, door.boundBox, color, thickness)
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
                        label_colors=['', 'blue', 'red', 'green', 'brown']
                        point=canvas.create_oval(pointx-5, pointy-5, pointx+5, pointy+5, fill=label_colors[label], tags='points', activefill=rgb2hex(*color), outline="")
                        canvas.tag_bind(point,"<Button-1>", lambda event, ind=ind: self.show_options(event, [canvas, door.ids, ind]))
                        #TODO ieladet jau izveidotos keypointus
 
    def show_options(self, *args):
        event=args[0]
        pressed_id = event.widget.find_closest(event.x, event.y)[0]
        selected_key=args[1]
        rcmenu = Menu(self.master, tearoff=0)
        rcmenu.add_command(label='kreisa prieksa', command = lambda:self.selected_menu_options(1, pressed_id, selected_key))
        rcmenu.add_command(label='kreisa aizmugure', command = lambda:self.selected_menu_options(2, pressed_id, selected_key))
        rcmenu.add_command(label='laba prieksa', command = lambda:self.selected_menu_options(3, pressed_id, selected_key))
        rcmenu.add_command(label='laba aizmugure', command = lambda:self.selected_menu_options(4, pressed_id, selected_key))
        rcmenu.post(event.x_root, event.y_root)

    def selected_menu_options(self, *event):
        label_colors=['', 'blue', 'red', 'green', 'brown']
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

    def draw(self, canvas, wwidth, wheight):
        
        canvas.delete("all")
        img = cv2.imread(self.segm_path)
        resized=self.resize(img, wwidth, wheight)
        resized=PIL.Image.fromarray(resized)       
        canvas.image=PIL.ImageTk.PhotoImage(resized)
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')
        self.drawDoors(canvas)
