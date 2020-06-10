class TapDetector():
    def __init__(this,channels,channels2,rate,index,index2,frames,chunk,tapSize):
        this.index2 = index2
        this.audio = pyaudio.PyAudio()
        this.freqFrames = np.full((tapSize,chunk),None,dtype='float64')
        this.freqFrames2 = np.full((tapSize,chunk),None,dtype='float64')
        this.frames = np.zeros(tapSize)
        this.all = []
        this.all2 = []
        this.tapSize = tapSize
        this.currentFrame = 0
        this.chunk = chunk
        this.rate = rate
        this.cooldown = 0
        this.trainingSets = []
        this.testSets = []
        this.negativeTrainingSet = []
        this.clf = None
        this.clfs = [("KNN",KNeighborsClassifier(),{'n_neighbors':range(2,10)}),
               ("DecisionTreeClassifier",DecisionTreeClassifier(),{'max_depth':range(2,10)}),
               ("SVC",SVC(),{'C':range(1,10),'kernel':('linear','rbf','poly','sigmoid')}),
               ("RandomForest",RandomForestClassifier(),{'max_depth':range(2,10), 'n_estimators':range(2,10), 'max_features':range(2,10)}),
               ("MLP",MLPClassifier(),{'alpha':range(1,5), 'max_iter':[2000]})
               ]
        this.classifier = KNeighborsClassifier(5)
        this.stream = this.audio.open(format = pyaudio.paInt16,
                                      channels = channels,
                                      rate = rate,
                                      input = True,
                                      input_device_index = index,
                                      frames_per_buffer = frames)
        if (this.index2 != None):
            this.audio2 = pyaudio.PyAudio()
            this.stream2 = this.audio.open(format = pyaudio.paInt16,
                                           channels = channels2,
                                           rate = rate,
                                           input = True,
                                           input_device_index = this.index2,
                                           frames_per_buffer = frames)
        #FILE READING
        this.wavFile = None
        this.wavFile2 = None
    
    def start(this,seconds):
        print("Recording Started")
        print(not this.clf == None)
        for i in range(0,int(this.rate / this.chunk * seconds)):
            if this.record():
                if not this.clf == None:
                    flat = this.byteTapDataHandler(this.getFreqFrames())
                    prediction = this.clf.predict([flat])
                    print(prediction)
        print("Recording Finnished")
    
    def stop(this):
        this.stream.stop_stream()
        this.stream.close()
        this.audio.terminate()
        
    def tapDataHandler(this,tap):
        mic1 = tap
        mic2 = None
        if not this.index2 == None:
            mic1 = tap[:len(tap)//2]
            mic2 = tap[len(tap)//2:]
        micTrans = this.tapTransform(mic1)
        if not this.index2 == None:
            micTrans.extend(this.tapTransform(mic2))   
        return micTrans
    
    def tapTransform(this,tap):
        filtered = list(map(lambda x: x if x > 2000 or x < -2000 else 1,tap))
        centered = this.centerTap(filtered)
        return centered
        f, t, Sxx = scysig.spectrogram(np.array(centered), len(centered))
        x,y = Sxx.shape
        Sxx = np.reshape(Sxx, (y, x))
        Sxx = np.array([s/max(s) for s in Sxx])
        return Sxx.flatten().tolist()
        
    def byteTapDataHandler(this,tap):
        decoded = this.byteArrayToTap(tap)
        return this.tapDataHandler(decoded)
    
    def byteArrayToTap(this,tapBytes):
        tapArray = []
        for byte in tapBytes:
            tapArray.extend(np.frombuffer(byte, dtype='<i2'))
        return tapArray
    
    def centerTap(this,tap):
        return tap
        tap = np.array(tap).tolist()
        startId = 0
        for freqId in range(0,len(tap)):
            if tap[freqId] > 5000:
                startId = freqId
                break
        startId += 600
        empty = np.zeros(len(endId)-(endId-startId))
        centered = np.hstack([np.array(tap[startId:]),empty])
        return centered[:len(centered)-700]    
    
    def save(this,fileName,fileName2):
        this.saveClip(fileName,this.all)
        this.saveClip(fileName2,this.all2)

    def saveClip(this,fileName,clip):
        waveFile = wave.open("{}.wav".format(fileName), 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(this.audio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(this.rate)
        waveFile.writeframes(b''.join(clip))
        waveFile.close()
    
    def read(this):
        if this.wavFile == None:
            data =this.stream.read(this.chunk, exception_on_overflow=False)
            if (this.index2 != None):
                data2 =this.stream2.read(this.chunk, exception_on_overflow=False)
        else:
            data = this.wavFile.readframes(this.chunk)
            if (this.index2 != None):
                data2 = this.wavFile2.readframes(this.chunk)
        res = data, None
        if (this.index2 != None):
            res = data, data2
        return res
        
    
    def detectFromFile(this,fileName,fileName2):
        this.wavFile = wave.open(fileName, 'rb')
        if (this.index2 != None):
            this.wavFile2 = wave.open(fileName2, 'rb')
        
    def detectFromMic(this,channels,channels2,index,index2,frames):
        this.wavFile = None
        this.stream = this.audio.open(format = pyaudio.paInt16,
                                              channels = channels,
                                              rate = this.rate,
                                              input = True,
                                              input_device_index = index,
                                              frames_per_buffer = frames)
        if (this.index2 != None):
            this.wavFile2 = None
            this.stream2 = this.audio2.open(format = pyaudio.paInt16,
                                            channels = channels2,
                                            rate = this.rate,
                                            input = True,
                                            input_device_index = index2,
                                            frames_per_buffer = frames)
            
    def record(this):
        tapFound = False

        data,data2 = this.read()
        if(len(data)<1):
            return tapFound
        decoded = np.frombuffer(data, dtype='<i2')
        this.all.append(data)
        this.frames[this.currentFrame] = audioop.rms(data, 2)
        this.freqFrames[this.currentFrame] = (decoded)
        if (this.index2 != None):
            decoded2 = np.frombuffer(data2, dtype='<i2')
            this.all2.append(data2)
            this.freqFrames2[this.currentFrame] = (decoded2)
            
        this.currentFrame = (this.currentFrame+1)%this.tapSize
        if(this.cooldown <= 0):
            tapFound = this.isTap(9000)
            return tapFound
        else:
            this.cooldown -= 1
        return tapFound
    
    #Makes the training sets by prompting and recording tap sounds
    def makeTrainingSet(this,buttons,tapsNeeded):
        for button in range(0,buttons):
            this.trainingSets.append([])
            print("\nCallibrating button position {}".format(button))
            for tap in range(1,tapsNeeded+1):
                while(not this.record()):
                    d = 1
                this.trainingSets[button].append(this.getFreqFrames())
                sys.stdout.write("\r{}/{} taps registered for button {}".format(tap,tapsNeeded,button+1))
                sys.stdout.flush()
     
    #Makes the test set by recording taps used for testing
    def makeTestSet(this,buttons,tapsNeeded):
        for button in range(0,buttons):
            this.testSets.append([])
            print("\nTesting for button position {}".format(button))
            for tap in range(1,tapsNeeded+1):
                while(not this.record()):
                    d = 1
                this.testSets[button].append(this.getFreqFrames())
                sys.stdout.write("\r{}/{} taps registered for button {}".format(tap,tapsNeeded,button))
                sys.stdout.flush()
    
    #runs the training set against a set of classification algorithms
    #prints out the confusion matrix for the best scoring algorithm
    def runTest(this):
        positiveTrainX, positiveTrainY = [], np.array([])
        for button in range(1,len(this.trainingSets)+1):
            positiveTrainX.extend([this.byteTapDataHandler(tap) for tap in this.trainingSets[button-1]][:10])
            positiveTrainY = np.hstack([positiveTrainY,np.full( len(this.trainingSets[button-1][:10]), button)])

        bestScore= 0
        bestClf = None
        clfs = this.clfs
        for clfId in [2]: #range(0,len(clfs)):
            name,clf,params = clfs[clfId]
            search = GridSearchCV(clf,params,refit=True)
            search.fit(positiveTrainX,positiveTrainY)
            print("{}:{}".format(name,search.best_score_))
            if(bestScore <= search.best_score_):
                bestScore = search.best_score_
                bestClf = search.best_estimator_
        print("Best clf:{}({})".format(bestClf,bestScore))
                
        this.clf = bestClf
    
    def getConfusionMatrix(this):
        TestX, TestY = [], np.array([])
        for button in range(1,len(this.trainingSets)+1):
            TestX.extend([this.byteTapDataHandler(tap) for tap in this.testSets[button-1]])
            TestY = np.hstack([TestY,np.full( len(this.testSets[button-1]), button)])
        plot_confusion_matrix(this.clf, TestX, TestY,cmap=plt.cm.Reds,normalize='true')
        plt.show()
        
    # Trains the classifier with the trainings set
    def fit(this):
        positiveTrainX, positiveTrainY = [], np.array([])
        for button in range(1,len(this.trainingSets)+1):
            setData = [this.byteTapDataHandler(tap) for tap in this.trainingSets[button-1][:]]
            positiveTrainX.extend(setData)
            positiveTrainY = np.hstack([positiveTrainY,np.full( len(this.trainingSets[button-1][:]), button)])
        trainX = positiveTrainX
        trainY = positiveTrainY
        this.clf = this.classifier.fit(trainX,trainY)

    #Prints the average frequency plot out for each training set
    #Used for debugging, and is now depricated
    def PrintButtonFreq(this):
        plots = []
        avgFreq = []
        avgTap = []
        print(np.array(this.trainingSets).shape)
        for tset in this.trainingSets:
            avgTap = this.tapDataHandler(tset[0])
            for tap in tset:
                #tap = this.centerTap(tap)
                avgFreq = tap[0]
                power = (np.abs(np.fft.rfft(avgTap)))
                freqs = (np.linspace(0, this.rate, len(avgTap)//2+1))
                plt.plot(freqs,power)
                plt.show()
                for freq in tap:
                    avgFreq += freq
                avgFreq = avgFreq/len(tap)
                avgTap += this.tapDataHandler(tap)
            avgTap = avgTap/len(tset)
            #plt.plot(avgTap)
            #plt.show()
            power = (np.abs(np.fft.rfft(avgTap)))
            plots.append(power)
            freqs = (np.linspace(0, this.rate, len(avgTap)//2+1))
            plt.plot(freqs,power)
            plt.show()
        print(len(plots))    
        plt.plot(np.linspace(0, this.rate, len(avgTap)//2+1),plots[0]-plots[3])
        plt.show()
        plt.plot(gaussian(100,2.5))
        plt.show()
    
    def isTap(this,threshold):
        soundSlice = np.hstack([this.frames[this.currentFrame:],this.frames[:this.currentFrame]])
        diff = np.diff(soundSlice)
        maxPeaks = argrelextrema(soundSlice, np.greater)
        if len(maxPeaks) > 0:
            maxPeaks = maxPeaks[0].tolist()
        if len(maxPeaks) == 0:
            return False
        if soundSlice[maxPeaks[0]] > threshold and diff[maxPeaks[0]]**2 > 5000:
            for i in range(1,len(maxPeaks)):
                if soundSlice[maxPeaks[i]] > (soundSlice[maxPeaks[0]] * 0.5):
                    return False
            this.saveClip("isTapTest",this.all[len(this.all)-50:])
            this.cooldown = this.tapSize
            return True
        return False
    
    def getFreqFrames(this):
        inOrder = this.all[len(this.all)-this.tapSize:]
        if (this.index2 != None):
            inOrder.extend(this.all2[len(this.all2)-this.tapSize:])
        return inOrder
                      
channels =1
rate = 44100
index = 1
index2 = 3
frames = 5
chunk = 1024//5
tapSize = 12
detector = TapDetector(channels,channels,rate,index,index2,frames,chunk,tapSize)
detector.detectFromFile("mic1Train4x200.wav","mic2Train4x200.wav")
detector.makeTrainingSet(4,200)
detector.detectFromFile("mic1Train4x100.wav","mic2Train4x100.wav")
detector.makeTestSet(4,100)
detector.runTest()
detector.getConfusionMatrix()
#detector.start(10)
#detector.save("mic1Test4x50","mic2Test4x50")
detector.stop()
