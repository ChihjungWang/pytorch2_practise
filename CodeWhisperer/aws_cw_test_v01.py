# write a method to download high resolution youtube videos and save them in a folder
import pytube
import os
import boto3

def download_video(url, folder):
    video = pytube.YouTube(url)
    stream = video.streams.first()
    stream.download(folder)




'''
import pytube

def download_video(url, folder):
    video = pytube.YouTube(url)
    stream = video.streams.first()
    stream.download(folder)

url = 'https://www.youtube.com/watch?v=wXcyq2Ay4uE'
floder = 'D://'
download_video(url, floder)

'''
