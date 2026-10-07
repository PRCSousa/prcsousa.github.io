---
layout: project
title: "Training a VLA model to move a robotic arm with my own hands (literally)"
tags: [VLA, python, reinforcement learning, computer vision]
start_date: 2026-10-03
end_date: 2026-10-09
ongoing: false
img_path: "/assets/project_images/humanoid_challenge/sim2real.gif"
description: "One week, one phone camera, 6GB of VRAM and a challenge."

---

# Training a VLA model to move a robotic arm with my own hands (literally)

In this project, I fine-tuned a VLA model using my own collection of videos and instructions (like "move left" or "rotate clockwise") with the objective of manipulating a robotic arm on custom instructions that I created, with my hand as a the teacher.

## How it began

My background as a researcher is in reinforcement learning, and naturally, I've always kept up with the current advancements in the domain, being the most interesting one to me VLAs. I feel like this technology, besides the wonders of an LLM, is the first actual 'sci-fi'-esque technology that will allow a computer to come to life and produce general-purpose machines.

Thing is, I've never built one, between finishing my Master's and working as a researcher, [Humanoid's](https://thehumanoid.ai/) provided me with the perfect environment to experiment and learn a bit more about this. Their challenge was to drive a robotic arm in a simulation env, and the main constraint was for us to use data that was collected by ourselves to do so. Time to get to work.

### The Plan

The idea behind what I will be doing is simple:
1. Create a set of simple instructions of things I want to make the robotic arm do.
2. Record myself doing it with my hand.
3. Convert hand into robot actions.
4. Replay what my hand did, this time on the simulator.
5. Fine-tune a VLA model with this data.

## Data Collection

This was the most straight-forward part of the project (at least the collection itself, we'll get to that). After some iterations of what I wanted to train on, I collected approximately 30 videos of me doing 9 different actions, slightly changing initial conditions to add some variety, you can see the data in [HuggingFace](https://huggingface.co/datasets/ReAscalon/humanoid_move_thing).

Some instructions include:
- Opening and closing the gripper;
- Rotating clockwise and counterclockwise;
- Moving in all directions;
- Holding still;

Observe some clapping:

![Some clapping](/assets/project_images/humanoid_challenge/clapping_hands.gif)

In total, I got approximately 10 videos for each instruction, to a total of 89 clips of 5-15 seconds.

## Hand Tracking

The example on the original post used AprilTags as a visual landmark to align the simulation and the video fields, allowing us to easily calculate the relative positions between the camera and the robotic arm.

As I though about the challenges behind embodied AI, I deliberatedly chose to not use any kind of field alignment technique, and instead derive everything I needed from the environment itself (specifically, my own hand as the reference). This, of course, comes with a huge challenge, which no questions asked, was the biggest hurdle that I faced when developing this idea.

### Converting video to numbers

With these videos in hand (ha!), now I'd have to convert whatever my hand was doing into some useful data. After some research, [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), a library made specifically for image recognition, allowed us to collect positional information on 21 landmarks throughout our hand and wrist, as seen here:

![hand_connections](/assets/project_images/humanoid_challenge/hand_connections.png)

And after a bit of trial and error, I got this going:

![tracked_clapping](/assets/project_images/humanoid_challenge/tracked_clapping.gif)

Fantastic. Note the line connecting the thumb and index, this will be the main way of indicating if I am gripping something or not, and here is where the biggest challenge began.

### Going forwards, and going upwards

Going from 2D video to 3D motion is quite hard, in the end we are operating with one dimension less than we need. Pixels move in two axes on a 2D video, and this does not provide us with any sense of depth, so moving my hand forwards produces the same change as rising it, a positive change in the Y axis, making 3D movement ambiguous in that scenario.

Going back to the original example, the use of AprilTags in the table allows us to calculate the geometry that locates the camera position relative to the table, and from here we can calculate coordinates and motion analytically through the change in video in relation to the camera position. And again, my objective was to do this without these markers, the challenge gave a lot of focus on data, so I wanted data to be my main concern.

My first instinct was to use MediaPipe's provided depth estimation, and after trying it for a while and seeing what kind of data it produced, I abandoned it, it was failing to distinguish between Y movement and Z movement. So, I tried to use my hand as a scale.

By making some assumptions in my hand size, I could use the distance between landmarks as a method to estimate depth, if my hand gets smaller, the landmarks get closer, which means the hand is further away. It did indeed work, now movement was being considerably better in terms of distinguishing both axes, but still nowhere near optimal. The problem in this approach wasn't all that opaque either, as hand position and perspective plays a huge role in this kind of calculations:

![perspective](/assets/project_images/humanoid_challenge/perspective.png)


## The stroke of genius

So I had to search for something a bit more intricate, and that's when I read about Perspective-n-Point (PnP).

Normally, PnP is used to estimate a camera's position using a set of known 3D points, and their corresponding 2D projections, and this is really how AprilTags work in the end, we know the physical size of the tag and where their corners sit, and we have their 2D video projections, so we can estimate the camera position.

![PnP](/assets/project_images/humanoid_challenge/PnP.png)

But, if we look at what we need to solve this problem, we actually (kinda) have everything we need to solve for it. Our hands, in terms of scale, do not really change all that much, and the palm specifically can't bend like our fingers, the palm is a rigid body, and we can estimate roughly their 3D positions (if we assume the palm landmarks are all coplanar to each other).

So, instead of calculating the relative position of the camera to our palm, we instead calculate the relative position of the palm to our camera!


![3dpalmplane](/assets/project_images/humanoid_challenge/3dpalmplane.gif)

This did work! And honestly in a much better way than I was expecting. As seen in the gif the axes are a bit crooked, in the end I assume the palm to be a plane, and this would be a limitation as any task requiring twisting my hand would throw this off, but for a hand serving as a proxy of a robotic arm, this will suffice.


### Preparing our data

Now that we can convert 2D video to hand positions, we will make use of the change in position of my hand to produce data, so my hand movement will be expressed as [dx, dy, dz, drx, dry, drz, gripper].

Due to the naturally jitter of human motion, and when I looked at the initial retargetings inside the simulator, the robotic arm shaked a lot, and this would introduce a lot of problems in terms of precision if I were to train the VLA on larger tasks where precision was key. So, across all the data, I computed an EMA to smooth everything out.

Another important change was calculating the hand position in relation to the camera angle, so by using an approximation of how tilted the camera was, we could calculate the rotation matrix to align our data coordinates to the coordinates of the simulator (aligned in relation to the table).

In terms of the gripper, initially I looked rougly at what would be decent values to delineate if my thumb-index distance meant opened or closed, but this was very unstable, so I opted to calculate a kmeans per video to find a decent threshold between setting the gripper as opened or closed. In hindsight, I assumed the gripper to be binary, which led me to try and find the threshold in the continuous data, when I could've just map the max and min of each clip to [-1, 1] respectively.

Finally, now we just had to convert this data into LIBERO's convention, which was very straightforward and just required some scaling.

![sim2real](/assets/project_images/humanoid_challenge/sim2real.gif)

With this, we have hand movement being decently translated into the simulation environment. Running this over my whole set of videos, and associating each one to their specific instruction, we end up with our dataset to train the VLA on.

## Training

In terms of training, given our constraints in terms of data, and the fact that I have less than a week to train, and that my PC has only 6GB of VRAM, I could neither train a whole new VLA model, nor run a full fine-tune on SmolVLA. With this in mind, I set my goal on at least optimizing SmolVLA through LoRA. This would take a small toll of roughly 2GB of VRAM on my computer and run relatively fast, approximately 6 hours per 30000 steps. Given my time and compute constraints, I'll expect a trade-off in performance to at least get some results.

## Results

In order to confirm if the fine-tuned model learnt the instructions that were provided in the dataset, I produced a small evaluation script that kept track of:

 - Δx, Δy, Δz: The relative distance travelled in each respective axis relative to the starting point.
 - abs. path: The total distance travelled by the arm.
 - rY: The range of which the arm moved in the Y axis, in order to see if movements were mainly unidirectional or if it covered the full range of the arm.
 - rG: The range of which the gripper opened. This will have a small bias due to the thresholding method I implemented.
 - sY, sZ, sG: The total of times the sign changed in the respective measure. The main intent of these values is to see if the arm attempted to do the circular motions in the clockwise and counterclockwise instructions.

There was supposed to be a table here, but alas I don't know why the interpreter is failing, so please refer to the table [here.](https://github.com/PRCSousa/humanoid-challenge#results)

Here we can visualize ```move left```:

![move left](/assets/project_images/humanoid_challenge/move_left.gif)


And ```move right```:

![move right](/assets/project_images/humanoid_challenge/move_right.gif)


And ```counterclockwise```:

![clockwise](/assets/project_images/humanoid_challenge/clockwise.gif)


## Discussion

Overall, I believe this project to be a success. We managed to map hand videos into a VLA model to teach it primitive commands, without using any kind of reference anchor. This model successfully learned some of these primitive commands when compared to the baseline counterpart.

### Temporar Consistency

The directional primitives were the easiest to train and the ones that worked most cleanly, while the most complex ones like moving in a clockwise motion failed. I believe that, when compared to the unidirectional ones where the frame by frame data indicates the same motion, having moments where we are going left and then moments where we are going right causes confusion in the model. This should be due to the model's lack of memory, as if it does not know "where in the loop" it currently is, it cannot produce the corresponding next motion. As our data does not have any kind of temporal feature as we feed it in random batches, and the LoRA does not have any recurrent component, it was expected for these kinds of instructions to fail from the get-go.

### Composite Instructions

I also tested with composite instructions (e.g. seeing if the model could interpret "move left and move up"), but sadly it could not interpret these actions, which from what I've seen seems to be consistent with the overall [literature](https://arxiv.org/abs/2607.00351) on the topic, and personally, an interesting path to research.

### Evaluation Metrics

The evaluation I built, in terms of the metric I collected, is pretty simple, but it is a deliberate choice, as for directionality makes up most of the instructions I tried to teach the model. I thought path length and sign changes are enough to measure simple oscillatory behaviour like circling and waving, but in reality, these features do not properly demonstrate this behaviour. If I were to train on harder tasks, I'd also have to read and idealize a much stronger set of experiments and evaluation metrics, and better ways to measure circular motion.


### Limitations

I found out about this challenge with one week remaining, and this was the constraint that shaped all the decisions I took. I had thought about training the arm on picking and placing objects, and trying to do so without any kind of anchor would mean spending the better part of my time idealizing a way to do so in terms of data. I believed (and still do believe) that the most interesting and important problem is the one upstream of training a policy: "how do I turn simple videos recorded through my phone into learnable actions", and this led me to devote most of my time to the data.

In terms of things that I would've done differently, the most obvious that comes to my mind was not producing any kind of testing units to run before the training itself. Training time was my biggest bottleneck, and coming to my computer after a 6-8 hour training session just to see that some data was mislabeled or that I had mixed some of the axes making the "left" instruction mean "up" was not ideal, mainly because it meant that it took me 6 hours to make a 5 minute fix. In total it took me 6 runs to train a decent model, with 3 of them being due to small mistakes on previous steps.

Another thing I'd like to have done was to explore, yet again, other ways to use my videos with respect to the fine-tuning of the VLA model. I think that I did not use the VLA model to its fullest potential. Maybe if I had explored ways I could've used the language encoder to reach some kind of compositional understanding, or use the vision encoder in some way to estimate the world state (and explore the world model approach that was also recommended). There are a lot of things to try in such a complex model, and given the limitations, I couldn't chase them all. 

## Conclusion

With one week, one phone, 6GB of VRAM and a problem I've never tackled before, I answered "can I drive a robotic manipulator using a VLA by using my own hands?" and the answer is pretty positive.

It's not a perfect solution, but what's most important was what I took away from it, this being that the most important part of any kind of machine learning problem lies on the data, and computer vision and embodied AI are no exception. Despite this, I also wanted to explore more the VLA architecture itself and develop some solution that made use of its individual components, and not just train it.

In my opinion, solving the ambiguity of 2D-to-3D without relying on physical anchors was the biggest technical hurdle, and implementing the PnP geometry solver with the use of my hand palm became the technical centerpiece of the project, which I am really proud of.

Despite these compromises, successfully building a whole pipeline from raw phone videos to a functional VLA policy was a massive success.

Thanks to [Humanoid](https://thehumanoid.ai) for the challenge. It was a good week. You can check the project in my repository [here](https://github.com/PRCSousa/humanoid-challenge).