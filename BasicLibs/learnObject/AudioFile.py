# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2018/9/26
    Desc : 
    Note : 
'''


class AudioFile:
    def __init__(self, filename):
        if not filename.endswith(self.ext):
            raise Exception("Invalid file format")

        self.filename = filename


class MP3File(AudioFile):
    ext = 'mp3'

    def play(self):
        print('playing {} as mp3'.format(self.filename))


class WavFile(AudioFile):
    ext = 'wav'

    def play(self):
        print('playing {} as wav'.format(self.filename))


class OggFile(AudioFile):
    ext = 'ogg'

    def play(self):
        print('playing {} as ogg'.format(self.filename))


if __name__ == '__main__':
    ogg = OggFile('myfile.ogg')
    ogg.play()
