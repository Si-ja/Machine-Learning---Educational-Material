This is a simple repository for running Gilfoyle's Bitcoin alert system in python.

Original idea from the series Silicon Valley: https://www.youtube.com/watch?v=uS1KcjkWdoU&t

Structure of the repository:
> Gilfoyle's Bitcoin Warning Rig.ipynb - gives more details on some particular functions that are used in the creating of the alert.

> BitCoin.py - the code to execute and allow fot the alert system to work.

> Execute_BitCoin.bat - a .bat file that automatically starts the BitCoin.py file without the need to open the program through any text editor files.

> Bitcoin_log.txt - log of all values that bitcoin takes when tracked with our program.
> Log.png - Visual representation of the Bitcoin_log.txt data.

> SUFFER.wav - the audio file that plays as an allert.

> Settings.txt - presets to run the BitCoin.py file. Their meaning: **currency** - in what currency we operate (can be USD, EUR, GBP); **treshold** - value below which if bitcoin drops the alert will play; **track** - for how many minutes we want our program to run (it updates every minute, so it also means, how many checks of bitcoin values will be done in total before the program closes).
