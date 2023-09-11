import pypianoroll
import os
import matplotlib.pyplot as plt
import numpy as np


threshold = 40

artist = "Beethoven"
pth = os.path.join("MIDI Music",artist)
np_path = os.path.join(pth,f"new_vae_music_{artist}.npy")


A=np.load(np_path)
print(A.shape)
A = A[:,:1000]
print(A.shape)
plt.imshow(A,aspect="auto")
plt.show()

#A=A/A.max()*127
div = np.max(A,axis=0)
A = np.divide(A, div, out=np.zeros(A.shape,dtype=float), where=(div!=0)) * 127
A=A.T
#A=A/A.max()*127
plt.imshow(A,aspect="auto")
plt.show()

print(A)
for e1 in range(len(A)):
    for e2 in range(len(A[0])):
        if A[e1,e2]<threshold:A[e1,e2]=0
#A=A.astype('uint8')
print(A.max())
print(A.shape)

plt.show()


B=pypianoroll.Track(pianoroll = A)
B=pypianoroll.StandardTrack(name = 'test_conv2d_binary',pianoroll=B)
B=pypianoroll.Multitrack(resolution =2.5, tracks = [B])

C=pypianoroll.to_pretty_midi(multitrack = B)
C.write(os.path.join(pth,f'new_vae_music_{artist}.mid'))
B.binarize()
B.plot()
plt.show()
