---
layout: post
title: "expanding my domain"
tags: [python, machine learning, pytorch]
---

As promised, this site now has its own dedicated **[projects](/projects.html)** section. I'll try to make it as interactive as possible by addimg not just what I did but *(if possible)* some interesting images to go along and make it more digestible. Right now we can click on each card to read more about the work.


Also, I've begun the development of a new idea that came to my mind:

> When developing a new model architecture, it is sometimes doomed to fail right from the start. We end up wasting a lot of resources (and more importantly, time) training it, just to watch it collapse or plateau six hours later. It sucks, I've been there a dozen times, and you too probably. Usually, when we talk about evaluating models, we mean evaluating the final weights that originated from an architecture once we finally found one that works. **But what if we could evaluate the architecture itself before we even begin training?**

I've begun researching about methods and metrics that, through the properties of the common components of an architecture, such as `dense layers`, `convolutional blocks`, `self-attention` etc., it is possible to understand how the gradients will behave, the flow of information through along the layers, it's long term stability and even capacity to learn.

There is a lot of scientific research scattered arround, and **no real tool** that allows a researcher to use each and every applicable metric tailored to their new architecture to confirm if what they are creating can actually converge and learn, before burning through hundreds of hours of `GPU` time.