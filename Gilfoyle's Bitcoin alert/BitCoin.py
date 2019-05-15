#%%
def full_process(currency = "USD", treshold = 5000, track = 5):
    """Initiate the function to track the falls of the bitcoin.
    Curreny could be USD, EUR or GBR, default is USD.
    Treshold of the currency can be whatever you set it to. Currently it is 5000...money.
    Decide for how long you want to track the change. Default is set to 5 minutes.
    
    Update occurs only once a minute as the api updates that offten."""

    #Function can be executed many times, but with how we set the whole project
    #We do not really care about what the default settings are. That will be adjusted
    #With our "Settings.txt" file. But you can reuse it in other way.
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import style
    import numpy as np
    from scipy.interpolate import spline
    import time
    import warnings
    from IPython.display import clear_output

    #Ignore the warnings if they come up
    #Just to not overclog our screen, but for reference:
    #Most warnings are about the scipy.interpolate.spline being depreciated
    #I think that's the word...it works enough for us now, let's not bother :3
    warnings.filterwarnings("ignore")
    
    currency = currency
    treshold = treshold
    track = track
    
    def clean_log():
        """This will initiate cleaning of the log file for tracking of the information from current time. 
        Done only 1 time per session."""
        open('Bitcoin_log.txt', 'w').close()
        
    def bitcoin(currency = "USD"):
        """Retrieve data from coindesk api when called. Can work on USD, EUR & GBP. By default is set to USD."""
        import requests
        currency = currency
        url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        response = requests.get(url)
        value = response.json()["bpi"][currency]["rate_float"]
        return value 
    
    def log_builder(data_money, instance):
        """Function that writes indicated data into a file by appending it.
        data_money - information about the current value of the bitcoin.
        instance - instance of the bitcoin tracked. Starts from 1 in the grand loop when used."""
        array_to_text = str(instance) + "," + str(data_money) + "\n"
        outF = open("Bitcoin_log.txt", "a")
        outF.writelines(array_to_text)
        outF.close()
        
    def ear_pain():
        """Initiate the melody when called."""
        import winsound
        winsound.PlaySound("SUFFER", winsound.SND_FILENAME)
    
    #Now we will initialize the creation of the graph.
    def animate():
        style.use("dark_background")
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        dipping_point = treshold

        graph_data = open("Bitcoin_log.txt", "r").read()
        lines = graph_data.split("\n")
        xs = []
        ys = []
        dipper = []

        itterator = len(lines) * 10
        for line in lines:
            if len(line) > 1:
                x, y = line.split(",")
                xs.append(float(x))
                ys.append(float(y))
                dipper.append(float(dipping_point))
        xs = np.array(xs)
        ys = np.array(ys)   

        x_smoth = np.linspace(xs.min(), xs.max(), itterator)
        dipper = np.linspace(dipping_point, dipping_point, itterator)
        y_smoth = spline(xs, ys, x_smoth)

        ax1.clear()   
        ax1.plot(xs, ys, "o")
        ax1.plot(x_smoth, y_smoth, 'c')
        ax1.axhline(y=dipping_point,  color='r', linestyle='-')
        ax1.fill_between(x_smoth, y_smoth, dipping_point, where = (y_smoth > dipper), color = "g", alpha = 0.6)
        ax1.fill_between(x_smoth, y_smoth, dipping_point, where = (y_smoth < dipper), color = "r", alpha = 0.6)
        ax1.set_title("Rate of Bitcoin in " + str(currency))
        ax1.set_ylabel("Price")
        ax1.set_xlabel('Instance')
        
    #Now we are only left with initiating of our unholly creature
    #And running it for as many minutes as we have indicated
    for instance in range(track):
        
        #clean our file once
        if instance == 0:
            clean_log()
            
        #Get our bitcoin value
        money = bitcoin(currency)
        #save it into the txt file
        log_builder(money, instance+1)

        #check if we are bellow our treshold and kindly inform us of it with a quiet sound:

        #---------------------------------%%%%%%%%%%%%%%%%%%%-------------------------------
        #Feature scratched as it does not make sense:
        #Check if the previous instance was also beyond the treshold not to initiate the sound
        #on every time we are lower than the treshold, only when we dip back again.

        #It does not make sense as: 
        #If we are only informed when the bitcoin is under our treshold, then how do we know when it's above it?
        #Exactly...there is no way. So effectivelly, we will only hear the sound when it dips, but if it runs in the
        #backgroun we will only know when it's dipping. If the sound is either played or not played every minute -
        #Then we know that for that minute the bitcoin is above our tresholder or not. 

        #This is the same situation in the series. Without either hearing the bitcoin going up after dipping bellow some point,
        #Or visualy inspecting it, you will only hold knowledge that the bitcoin is always under a certain value.
        #Keeping the code though if it might inspire some people to come up with a solution for this dilema.

        #history_check = open("Bitcoin_log.txt", "r")
        #lineList = history_check.readlines()
        #history_check.close()
        #second_final_line = float(lineList[len(lineList)-2].strip().split(",")[1])

        if money < treshold:
            ear_pain()
            
        #Produce and save the graph after at least 2 itteration
        #For 1 and 2 in general - the graph will not make much sense
        #Nor can it be properly drawn
        #For the 3rd itteration it's also quite bad, so really
        #It only starts making sesne from the fourth itteration.
        if instance != 0 and instance != 1:
            animate()
            plt.savefig("Log.png")
        #and wait for a bit before the next instance is taken...60 seconds to be exact
        time.sleep(60)

#Currently we can indicate our parameters in the settings file
#And execute everything with a single bash file.
#You know...FOR KIDS! To make it more simple to be honest.
settings_data = open("Settings.txt", "r").read()
info = settings_data.split("\n")
currency  = str(info[0][11:].strip())
treshold = float(info[1][11:].strip())
track = int(info[2][8:].strip())

full_process(currency = currency, treshold = treshold, track = track)