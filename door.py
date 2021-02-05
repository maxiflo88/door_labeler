class door:
    
    def __init__(self, ids, boundBox=None, keypoints=[]):
        self.ids=ids
        self.boundBox=boundBox
        self.keypoints=keypoints

    def keypointFormat(self):
        keysFormated=np.zeros((4*3), dtype=np.int32)
        num_keypoints=0
        for keypoint in self.keypoints:
            if keypoint[2]==1:
                keysFormated[:3]=[*keypoint[:2], 2]
                num_keypoints+=1
            if keypoint[2]==2:
                keysFormated[3:6]=[*keypoint[:2], 2]
                num_keypoints+=1
            if keypoint[2]==3:
                keysFormated[6:9]=[*keypoint[:2], 2]
                num_keypoints+=1
            if keypoint[2]==4:
                keysFormated[9:12]=[*keypoint[:2], 2]
                num_keypoints+=1
        return keysFormated.tolist(), num_keypoints

    def addKeypointLabel(self, keypoint_id, label):
        ksize=len(self.keypoints)
        for i in range(ksize):
            if self.keypoints[i][2]==label:
                self.keypoints[i][2]=0
        self.keypoints[keypoint_id][2]=label

    @property
    def keypoints(self):
        return self.__keypoints

    @keypoints.setter
    def keypoints(self, values):
        if not values:
            self.__keypoints=[]
        elif len(values[0])==3:
            self.__keypoints=values
        else:
            for value in values:
                self.__keypoints.append([*value, 0])