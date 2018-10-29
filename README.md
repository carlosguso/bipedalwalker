# bypedalwalker
Bipedal Walker env from OpenAI enviroment. Midterm assignment for Move37 from theshool.ai

# Augmented Random Search
This is a state of the art algorithm that helps us find the optimal policy way more faster that other algorithms e.g. Q-learning
This algorithm is explain in detail by [colin skow](https://github.com/colinskow/move37/tree/master/ars) ands is part of Move37,
a [theschool.ai](https://www.theschool.ai) course.
I try the algorithm and it starts to converge at the 300 step aprox., but i wanted to be able to save the values of the weight matrix, so
since the ars.py is using Numpy to create the matrix, we used a built-in function to save it into a csv so we can use this values later to achive
similar results. 
