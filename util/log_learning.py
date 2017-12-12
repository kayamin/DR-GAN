#!/usr/bin/env python
# encoding: utf-8

def log_learning(epoch, steps, modelname, loss, args, D_precision=None):
    text = "EPOCH : {0}, step : {1}, {2} : {3}, precision : {4}".format(epoch, steps, modelname, loss, D_precision)
    print(text)
    with open('{}/Learning_Log.txt'.format(args.save_dir),'a') as f:
        f.write("{}\n".format(text))
