#!/usr/bin/env python
# encoding: utf-8

def log_learning(epoch, steps, modelname, loss, args):
    text = "EPOCH : {0}, step : {1}, {2} : {3}".format(epoch, steps, modelname, loss)
    print(text)
    with open('{}/Learning_Log.txt'.format(args.save_dir),'a') as f:
        f.write("{}\n".format(text))
