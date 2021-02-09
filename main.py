from tkinter import *
import numpy as np
from PIL import ImageTk, Image, ImageDraw, ImageColor
from tkinter import filedialog
from glob import glob
import os
from image import image
class Paint(object):
    #TODO izveidot zoom f-iju
    #TODO izveidot lai punktus var ielikt jebkura vieta
    def __init__(self, master):
        self.master=master
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=2)
        
        self.master.rowconfigure(1, weight=1)
        self.images=[]

        self.Save_button = Button(self.master, text='Save image', command=self.save)
        self.Save_button.grid(row=0, column=5, sticky="NSEW")

        self.load_button = Button(self.master, text='Load image', command=self.openfn)
        self.load_button.grid(row=0, column=4, sticky="NSEW")

        self.detect_doors_button = Button(self.master, text='detect doors', command=self.detect_doors)
        self.detect_doors_button.grid(row=0, column=3, sticky="NSEW")

        self.detect_keypoints_button = Button(self.master, text='detect keypoints', command=self.detect_keypoints)
        self.detect_keypoints_button.grid(row=0, column=2, sticky="NSEW")

        self.img_prev_button = Button(self.master, text='Prev IMG', command=self.use_img_prev)
        self.img_prev_button.grid(row=2, column=4, sticky="NSEW")

        self.img_next_button = Button(self.master, text='Next IMG', command=self.use_img_next)
        self.img_next_button.grid(row=2, column=5, sticky="NSEW")

        self.canvas = Canvas(self.master)
        self.canvas.grid(row=1, columnspan=5, sticky="NSEW")
        self.drawImage()
        
    def save(self):
        image=self.images[self.img_index]
        image.save()

    def detect_doors(self):
        image=self.images[self.img_index]
        image.detect_doors()
        self.drawImage()

    def detect_keypoints(self):
        image=self.images[self.img_index]
        image.detect_keypoints()
        self.drawImage()


    def use_img_next(self):
        if self.images:
            self.img_index+=1
            if self.img_index>len(self.images)-1:
                self.img_index=0
            self.drawImage()

    def use_img_prev(self):
        if self.images:
            self.img_index-=1
            if self.img_index<0:
                self.img_index=len(self.images)-1
            self.drawImage()

    def openfn(self):
        dir_path = filedialog.askdirectory()
        self.img_index=0
        self.images=[image(imgPath, self.master) for imgPath in glob(os.path.join(dir_path,'*.png'))]
        self.drawImage()

    def drawImage(self):
        if self.images:
            image=self.images[self.img_index]
            self.master.winfo_toplevel().title(image.img_path)
            image.draw(self.canvas, self.canvas.winfo_width(), self.canvas.winfo_height())
	

if __name__ == '__main__':
    root = Tk()
    Paint(root)
    root.mainloop()