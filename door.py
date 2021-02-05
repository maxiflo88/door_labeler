import numpy as np
class door:
    
    def __init__(self, ids, boundBox=None, keypoints=[]):
        self.ids=ids
        self.boundBox=boundBox
        self.keypoints=keypoints

    def toCocoFormatKeypoints(self, labels):
        keysFormated=np.zeros((len(labels)*3), dtype=np.int32)
        num_keypoints=0
        for keypoint in self.keypoints:
            range1=0
            range2=3
            for label in labels: 
                if keypoint[2]==label['id']:
                    keysFormated[range1:range1+3]=[*keypoint[:2], 2]
                    num_keypoints+=1
                range1=range2
                range2+=3
        return keysFormated.tolist(), num_keypoints

    def fromCocoFormatKeypoints(self):
        '''
        Read list of keypoints from Coco format [x, y, v, x, y, v] to [[x, y, 1][ x, y, 2]]
        '''
        pass
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