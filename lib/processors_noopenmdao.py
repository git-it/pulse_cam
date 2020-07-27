import numpy as np
import time
import cv2
import pylab
import os
import sys
# from lib.interface import plotXY, imshow

# for displaying the output
import PIL
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):
        self.camera = None
        self.rootTK = Tk()
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        # self.trained = False

        self.idx = 1
        self.find_faces = True

        ## setup GUI
        self.rootTK.bind('<Escape>', lambda e: self.rootTK.quit())
        self.lmain = Label(self.rootTK)
        self.lmain.pack(side=tk.LEFT, fill=tk.BOTH)

        ################
        self.respRate_fig = plt.Figure(figsize=(6,5), dpi=100)
        self.respRate_ax1 = self.respRate_fig.add_subplot(111)
        self.respRate_graph = FigureCanvasTkAgg(self.respRate_fig, self.rootTK)
        self.respRate_graph.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        # df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
        # df1.plot(kind='bar', legend=True, ax=ax1)
        self.respRate_ax1.set_title('Respitory Rate')

        self.heartRate_fig = plt.Figure(figsize=(6,5), dpi=100)
        self.heartRate_ax1 = self.heartRate_fig.add_subplot(111)
        self.heartRate_graph = FigureCanvasTkAgg(self.heartRate_fig, self.rootTK)
        self.heartRate_graph.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        # df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
        # df1.plot(kind='bar', legend=True, ax=ax1)
        self.heartRate_ax1.set_title('Heart Rate')
        ###############

    # def shift(self, detected):
    #     x, y, w, h = detected
    #     center = np.array([x + 0.5 * w, y + 0.5 * h])
    #     shift = np.linalg.norm(center - self.last_center)

    #     self.last_center = center
    #     return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w/2),
                int(h * fh_h/2)]

    def get_subface_means(self, coord):
        '''
        get the mean value accross an area
        '''
        x, y, w, h = coord
        self.times.append(time.time() - self.t0) ## time of subface mean
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])
        self.data_buffer.append( (v1 + v2 + v3) / 3.0 )
        #return 

    # def train(self):
    #     self.trained = not self.trained
    #     return self.trained

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def pupulate_gui(self ):
        ### plot graphs
        # [[self.times,
        #          self.samples],
        #         [self.freqs,
        #          self.fft]],
        self.respRate_ax1.plot( self.times,self.samples) #, linestyle="None", marker='o'
        self.respRate_graph.draw()
        self.respRate_ax1.clear()
        #df1.plot(kind='bar', legend=True, ax=self.respRate_ax1)

        ### plot graphs
        self.heartRate_ax1.plot(self.freqs,self.fft )
        self.heartRate_graph.draw()
        self.heartRate_ax1.clear()
        #df1.plot(kind='bar', legend=True, ax=self.heartRate_ax1)

        #### camera with overlay ####
        cv2image = cv2.cvtColor(self.frame_out, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        # self.lmain.after(10, self.pupulate_gui)
        # self.rootTK.mainloop()
        self.rootTK.update_idletasks()

    def face_finder(self, conf_threshold=0.75 ):
        '''
        return all face detections from RNN
        '''
        # modelFile = "model/opencv_face_detector_uint8.pb"
        # configFile = "model/opencv_face_detector.pbtxt"
        col = (100, 255, 100)
        base_path = os.path.abspath(".")
        modelFile = os.path.join(base_path, "lib\model\opencv_face_detector_uint8.pb")
        configFile = os.path.join(base_path, "lib\model\opencv_face_detector.pbtxt")
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        frameHeight = self.frame_out.shape[0]
        frameWidth = self.frame_out.shape[1]
        blob = cv2.dnn.blobFromImage(self.frame_out, 1.0, (300, 300), [100, 100, 100], False, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            #print(detections);
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append(np.array([x1, y1, x2, y2]) )
                # bboxes.append(np.array([detections[0, 0, i, 3],detections[0, 0, i, 4],
                #         detections[0, 0, i, 5],detections[0, 0, i, 6] ] ) )
                cv2.rectangle(self.frame_out, (x1, y1), (x2, y2), (0, 255, 0), 
                            int(round(frameHeight/150)), 8)
                cv2.putText(self.frame_out, "Face",
                       (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, int(round(frameHeight/150)) )
        
        if len(bboxes) > 0:
            bboxes.sort(key=lambda a: a[-1] * a[-2])
            self.face_rect = bboxes[-1]

            # crap method to set forehead
            forehead1 = self.get_subface_coord(0.22, 0.13, 0.2, 0.1) 
            self.forehead1 = forehead1
            x, y, w, h = self.face_rect
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            # cv2.putText(self.frame_out, "Forehead",
            #             (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            ###
            ## get forehead info
            self.get_subface_means(forehead1)
            self.L = len(self.data_buffer)
            if self.L > self.buffer_size: # instead of >
                self.data_buffer = self.data_buffer[-self.buffer_size:]
                self.times = self.times[-self.buffer_size:]
                self.L = self.buffer_size
        
        #return bboxes
    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.times,
                 self.samples],
                [self.freqs,
                 self.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name="",
               bg=self.slices[0])

    def run(self, cam):
        # self.data_buffer, self.times, self.trained = [], [], False
        self.data_buffer, self.times = [], []

        # continous process instead
        while cv2.waitKey(1) < 0:
            try:
                self.frame_in = self.camera.get_frame()
            except:
                print("no camera yet")

            self.frame_out = self.frame_in
            self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                    cv2.COLOR_BGR2GRAY))
            col = (100, 255, 100)
            if self.find_faces:
                self.face_finder() #detected_new
                #return
            if set(self.face_rect) == set([1, 1, 2, 2]):
                return
            
            print('data_buffer', len(self.data_buffer))
            processed = np.array(self.data_buffer)
            self.samples = processed
            if self.L > 10 and self.buffer_size > 10:
                self.output_dim = processed.shape[0]
                self.fps = float(self.L) / (self.times[-1] - self.times[0])
                even_times = np.linspace(self.times[0], self.times[-1], self.L)
                interpolated = np.interp(even_times, self.times, processed)
                interpolated = np.hamming(self.L) * interpolated
                interpolated = interpolated - np.mean(interpolated)
                raw = np.fft.rfft(interpolated)
                phase = np.angle(raw)
                self.fft = np.abs(raw)
                self.freqs = float(self.fps) / self.L * np.arange(self.L / 2 + 1)

                freqs = 60. * self.freqs
                idx = np.where((freqs > 50) & (freqs < 180))

                try:
                    pruned = self.fft[idx]
                except:
                    continue
                phase = phase[idx]

                pfreq = freqs[idx]
                self.freqs = pfreq
                self.fft = pruned
                idx2 = np.argmax(pruned)

                t = (np.sin(phase[idx2]) + 1.) / 2.
                t = 0.9 * t + 0.1
                alpha = t
                beta = 1 - t

                self.bpm = self.freqs[idx2]
                self.idx += 1

                x, y, w, h = self.get_subface_coord(0.22, 0.13, 0.2, 0.1) #get_subface_coord(0.5, 0.18, 0.25, 0.15)
                r = alpha * self.frame_in[y:y + h, x:x + w, 0]
                g = alpha * \
                    self.frame_in[y:y + h, x:x + w, 1] + \
                    beta * self.gray[y:y + h, x:x + w]
                b = alpha * self.frame_in[y:y + h, x:x + w, 2]
                self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                            g,
                                                            b])
                x1, y1, w1, h1 = self.face_rect
                self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
                col = (100, 255, 100)
                gap = (self.buffer_size - self.L) / self.fps
                # self.bpms.append(bpm)
                # self.ttimes.append(time.time())
                if gap:
                    text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
                else:
                    text = "(estimate: %0.1f bpm)" % (self.bpm)
                tsize = 1
                cv2.putText(self.frame_out, text,
                        (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)
        
            # show the processed/annotated output frame
            #imshow("Processed", self.frame_out)
            # self.make_bpm_plot()
            self.pupulate_gui()