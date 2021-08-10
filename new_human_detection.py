
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
from matplotlib import pyplot as plt
import imutils
import cv2

enabled = False
 
class BitView27:
    def __init__(self, parent, title):
        self.parent = parent
         
        self.parent.geometry("600x600")
        self.parent.title(title)
        self.parent.protocol("WM_DELETE_WINDOW", self.onTutup)
        self.parent.resizable(False, False)
         
        self.aturKomponen()
         
    def aturKomponen(self):
        # buat main frame
        mainFrame = Frame(self.parent, bd=10)
        mainFrame.pack(fill=BOTH, expand=YES)
         
        # buat label penampil image
        self.lblImage = Label(mainFrame, text='Image Preview')
        self.lblImage.pack(side=TOP, fill=BOTH, expand=YES)
         
        # buat menu
        menuBar = Menu(self.parent)
         
        fileMenu = Menu(menuBar, tearoff=1)
        fileMenu.add_command(label='Open Image', underline=0, command=self.onBuka)
        fileMenu.add_command(label='Exit', underline=1, command=self.onTutup)
        menuBar.add_cascade(label='File', underline=0, menu=fileMenu)
        
        #added "file" to our menu
        algoritmaMenu = Menu(menuBar, tearoff=1)
        algoritmaMenu.add_command(label='Eye', underline=0, command=self.algoritmaHOG)
        algoritmaMenu.add_command(label='Upper Body', underline=6, command=self.algoritmaSVM)
        algoritmaMenu.add_command(label='Full Body', underline=0, command=self.algoritmaEdge)
        algoritmaMenu.add_command(label='Lower Body', underline=0, command=self.algoritmaCanny)
        menuBar.add_cascade(label='Algoritma', underline=0, menu=algoritmaMenu)
         
        self.parent.config(menu=menuBar)
         
    def onBuka(self, event=None):
        global enabled
        global filename
        
        #tipeFile = [('Python files', '*.py'), ('All files', '*')]
        filename = filedialog.askopenfilename(initialdir = "/home/pi/foto",title = "Select file",filetypes = (("jpeg files","*.jpeg"),("all files","*.*")))
             
        #panjang = 600
        #lebar = 450
         
        if filename:
            pict = Image.open(filename)
            gambar = ImageTk.PhotoImage(pict)
             
            self.lblImage.config(image=gambar)
            self.lblImage.image = gambar
             
            self.parent.title("Human Detection :: " + filename)
            
        
    def algoritmaHOG(self, event=None):
        image = cv2.imread(filename)
        cascade = cv2.CascadeClassifier("/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        rects[:,2:] += rects[:,:2]          
     
        orig = image.copy()

        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)

        rects = np.array([[x, y, w, h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
            
        # show the output image
        plt.suptitle("Algoritma Eye")
        plt.subplot(121),plt.imshow(orig, cmap='gray')
        plt.title('Before NMS : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image, cmap='gray')
        plt.title('After NMS : '+ str(len(pick))), plt.xticks([]), plt.yticks([])
        plt.show()
        
    def algoritmaSVM(self, event=None):
        image = cv2.imread(filename)
        cascade = cv2.CascadeClassifier("/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_mcs_upperbody.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        rects[:,2:] += rects[:,:2]
            
        orig = image.copy()

        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), ( w, h), (0, 0, 255), 2)

        rects = np.array([[x, y, w, h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
            
        # show the output image
        plt.suptitle("Algoritma Upper Body")
        plt.subplot(121),plt.imshow(orig, cmap='gray')
        plt.title('Before NMS : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image, cmap='gray')
        plt.title('After NMS : '+ str(len(pick))), plt.xticks([]), plt.yticks([])
        plt.show()
        
    def algoritmaEdge(self, event=None):
        image = cv2.imread(filename)
        cascade = cv2.CascadeClassifier("/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_fullbody.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        rects[:,2:] += rects[:,:2]
            
        orig = image.copy()

        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)

        rects = np.array([[x, y, w, h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
            
        # show the output image
        plt.suptitle("Algoritma Full Body")
        plt.subplot(121),plt.imshow(orig, cmap='gray')
        plt.title('Before NMS : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image, cmap='gray')
        plt.title('After NMS : '+ str(len(pick))), plt.xticks([]), plt.yticks([])
        plt.show()
    
    def algoritmaCanny(self, event=None):
        image = cv2.imread(filename)
        cascade = cv2.CascadeClassifier("/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_lowerbody.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

        rects[:,2:] += rects[:,:2]
            
        orig = image.copy()

        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)

        rects = np.array([[x, y, w, h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
            
        # show the output image
        plt.suptitle("Algoritma Lower Body")
        plt.subplot(121),plt.imshow(orig, cmap='gray')
        plt.title('Before NMS : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image, cmap='gray')
        plt.title('After NMS : '+ str(len(pick))), plt.xticks([]), plt.yticks([])
        plt.show()
          
    def onTutup(self, event=None):
        self.parent.destroy()
        plt.close()
    
         
if __name__ == '__main__':
    root = Tk()
     
    app = BitView27(root, "Human Detection")
     
    root.mainloop()
