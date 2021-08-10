
from tkinter import filedialog 
from tkinter import *
from PIL import Image, ImageTk
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
from matplotlib import pyplot as plt
import datetime
import imutils
import cv2

enabled = False

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

def compare_select(a,b):
    c = np.zeros(len(a))

    for i in range (len(a)):
        for j in range (len(a)):
            err = abs(a[i]-b[j])
            if err == 0.0 or err < abs(0.1):
                c[i] = b[j]
    return c
 
class mainMenu:
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
        algoritmaMenu.add_command(label='Multi Detect', underline=0, command=self.deteksiMulti)
        #algoritmaMenu.add_command(label='HOG', underline=0, command=self.algoritmaHOG)
        menuBar.add_cascade(label='Algoritma', underline=0, menu=algoritmaMenu)
         
        self.parent.config(menu=menuBar)
         
    def onBuka(self, event=None):
        global enabled
        global filename
        
        #tipeFile = [('Python files', '*.py'), ('All files', '*')]
        filename = filedialog.askopenfilename(initialdir = "/home/pi/foto",title = "Select file",filetypes = (("jpeg files","*.jpeg"),("all files","*.*")))
                     
        if filename:
            pict = Image.open(filename)
            gambar = ImageTk.PhotoImage(pict)
             
            self.lblImage.config(image=gambar)
            self.lblImage.image = gambar
             
            self.parent.title("Human Detection :: " + filename)
            
        
    def deteksiMulti(self, event=None):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        start = datetime.datetime.now()
        arr5 = np.array([])
        arr1 = np.array([])
        arr2 = np.array([])
        arr3 = np.array([])
        arr4 = np.array([])
        rects_all= np.array([])
        cnt_arr5 = 0
        cnt_arr1 = 0
        cnt_arr2 = 0
        cnt_arr3 = 0
        cnt_arr4 = 0
        image = cv2.imread(filename)
        cascade = cv2.CascadeClassifier("/home/pi/ines/haarcascade_frontalface_alt.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        image = cv2.imread(filename)
        cascade = cv2.CascadeClassifier("/home/pi/ines/haarcascade_profileface.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        orig1 = image.copy()
        orig = image.copy()
        if len(rects) > 0:
            rects[:,2:] += rects[:,:2]
            cnt_arr1 = len(rects)
                              
            for (x, y, w, h) in rects:
                label = str((datetime.datetime.now() - start).total_seconds())
                cv2.rectangle(orig1, (x, y), (w, h), (0, 0, 255), 2)
                cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)
                cv2.putText(orig1,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                c = (w - x) // 2
                pos_c = x + c
                arr1 = np.append(arr1, [pos_c])
                arr1 = selection_sort(arr1)
        
            rects_all = np.append(rects_all, arr1)
        #plt.suptitle("Haar Cascade\n"+ str(arr))
        plt.suptitle("Multi Detect")
        plt.subplot(231),plt.imshow(orig1, cmap='gray')
        plt.title('Muka : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        #plt.text(0.1, 0.1, "Data : \n"+ str(rects_all), fontsize=10)
        
        cascade = cv2.CascadeClassifier("/home/pi/ines/haarcascade_mcs_upperbody.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        orig2 = image.copy()
        if len(rects) > 0:
            rects[:,2:] += rects[:,:2]
            cnt_arr2 = len(rects)

            for (x, y, w, h) in rects:
                label = str((datetime.datetime.now() - start).total_seconds())
                cv2.rectangle(orig2, (x, y), ( w, h), (0, 0, 255), 2)
                cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)
                cv2.putText(orig2,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                c = (w - x) // 2
                pos_c = x + c
                arr2 = np.append(arr2, [pos_c])
                arr2 = selection_sort(arr2)
                
            rects_all = np.append(rects_all, arr2)
            
        plt.subplot(232),plt.imshow(orig2, cmap='gray')
        plt.title('Kepala Pundak : '+ str(len(rects))), plt.xticks([]), plt.yticks([])

        cascade = cv2.CascadeClassifier("/home/pi/ines/haarcascade_fullbody.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        orig3 = image.copy()        
        if len(rects) > 0:
            rects[:,2:] += rects[:,:2]
            cnt_arr3 = len(rects)

            for (x, y, w, h) in rects:
                label = str((datetime.datetime.now() - start).total_seconds())
                cv2.rectangle(orig3, (x, y), (w, h), (0, 0, 255), 2)
                cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)
                cv2.putText(orig3,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                c = (w - x) // 2
                pos_c = x + c
                arr3 = np.append(arr3, [pos_c])
                arr3 = selection_sort(arr3)
                
            rects_all = np.append(rects_all, arr3)
            
        plt.subplot(233),plt.imshow(orig3, cmap='gray')
        plt.title('Full Body : '+ str(len(rects))), plt.xticks([]), plt.yticks([])

        cascade = cv2.CascadeClassifier("/home/pi/ines/haarcascade_lowerbody.xml")
        rects = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(5, 5),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        
        orig4 = image.copy()
        if len(rects) > 0:
            rects[:,2:] += rects[:,:2]
            cnt_arr4 = len(rects)
            
            for (x, y, w, h) in rects:
                label = str((datetime.datetime.now() - start).total_seconds())
                cv2.rectangle(orig4, (x, y), (w, h), (0, 0, 255), 2)
                cv2.rectangle(orig, (x, y), (w, h), (0, 0, 255), 2)
                cv2.putText(orig4,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                c = (w - x) // 2
                pos_c = x + c
                arr4 = np.append(arr4, [pos_c])
                arr4 = selection_sort(arr4)
                
            rects_all = np.append(rects_all, arr4)
            
        plt.subplot(234),plt.imshow(orig4, cmap='gray')
        plt.title('Kaki : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
        
        orig5 = image.copy()
        if len(rects) > 0:
            cnt_arr5 = len(rects)
            for (x, y, w, h) in rects:
                label = str((datetime.datetime.now() - start).total_seconds())
                cv2.rectangle(orig5, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(orig5,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                c =( (x+w) - x) // 2
                pos_c = x + c
                arr5 = np.append(arr5, [pos_c])
                arr5 = selection_sort(arr5)
                
            rects_all = np.append(rects_all, arr5)
        
        plt.subplot(235),plt.imshow(orig5)
        plt.title('HOG : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        
        rects_all = selection_sort(rects_all)
        
        c = np.zeros(len(rects_all))
        
        for i in range (len(rects_all)-1):
            if (rects_all[i] == rects_all[i+1]):
                c[i] = 0
            elif ((rects_all[i] >= rects_all[i+1]- 5) and (rects_all[i] <= rects_all[i+1]+ 5)):
                c[i] = 0
            else:
                c[i] = rects_all[i]
            c[i+1] = rects_all[i+1]
            
        data_all = np.array([])
        for i in range (len(c)):    
            if c[i] != 0:
                data_all = np.append(data_all, c[i])
        
        str_arr = ''
        #str_arr = str_arr + '\n' + str(c)+ '\n' + str(rects_all)
        
            
        #pick = non_max_suppression(rects_all, probs=None, overlapThresh=0.65)
        #for (xA, yA, xB, yB) in pick:
        #    cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
        plt.subplot(236), plt.imshow(orig)
        plt.title('Total : '+ str(len(data_all))), plt.xticks([]), plt.yticks([])
        
        #plt.title('Data : '+ str_arr), plt.xticks([]), plt.yticks([])
        #plt.title('Data : '), plt.xticks([]), plt.yticks([])
        #plt.text(0.5, 0.5, 'boxed italics text in data coords', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        #plt.text(0.1, 0.1, 'Data : '+ str_arr, fontsize=10)
        
        plt.show()
        
    def algoritmaHOG(self, event=None):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        image = cv2.imread(filename)
        #image = cv2.medianBlur(img,3)
            
        #image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
        arr =np.array([])

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            c =( (x+w) - x) // 2
            pos_c = x + c
            arr = np.append(arr, [pos_c])
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.8)
    # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)
            
        # show the output image
        plt.suptitle("Algoritma HOG + SVM\n"+ str(arr))
        plt.subplot(121),plt.imshow(orig)
        plt.title('Before NMS : '+ str(len(rects))), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image)
        plt.title('After NMS : '+ str(len(pick))), plt.xticks([]), plt.yticks([])
        plt.show()
        
    
          
    def onTutup(self, event=None):
        self.parent.destroy()
        plt.close()
    
         
if __name__ == '__main__':
    root = Tk()
     
    app = mainMenu(root, "Human Detection")
     
    root.mainloop()

