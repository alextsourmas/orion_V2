* In order to use the trading services you have to set your alpaca keys with environment variables before running the script. In terminal:
`export ALPACA_KEY='<key id>'`
`export ALPACA_SECRET='<secret key>'`

*Once environment variables have been set, use the following command to execute nohup continuously:
`nohup python3 -u main.py &`
You'll then see a 'nohup.out' file created in the directory that will continously print the output of the script


* If for any reason you need to kill the nohup process from running, first do:
`lsof nohup.out`
to identify the PID running the nohup process, then 
`kill -9 <PID`
to kill it. You'll see a message from nohup indicating it's been killed

* **NOTE**: You can set the 'cash_ratio' to determine the amount of available funds to use for a purchase. It defaults to 1000 and is currently set to 100 on `main.py:792: schedule.every().day.at("19:32").do(job,'BTC-USD',100)`.
* You must set the scheduled time for the process to run as well on line 792 in military time. If you want it to run at 8:00 am, use "08:00"