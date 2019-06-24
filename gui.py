# import tkinter as tk
# win = tk.Tk()
# win.title("Python gui")
# win.mainloop()
#
# class TestBed:
#     def __init__(self):
#         return self
#
#     def attack1(self):
#         # attack the system
#
#     def attack2(self):
#         # attack the system
#
#

class Solution:
    def numPairsDivisibleBy60(self, time):
        n = 0
        ls = []
        for i in range(len(time)):
            # [30,20,150,100,40,50,10]
            # [10,20,30,40,50,100,150]
            #
            ls.append(time[i] % 60)

        d = {}
        for i in range(len(time)):
            if d.get(time[i], False):
                d[time[i]] += 1
            else:
                d[time[i]] = 1


        for i in range(len(time)):
            c = d[time[i]]
            if time[i] == 0 or time[i] == 30:
                n += (c * (c-1)) / 2
            else:
                n += c * d[time[60-i]]

        return n

k = Solution()
print (k.numPairsDivisibleBy60([30,20,150,100,40]))
