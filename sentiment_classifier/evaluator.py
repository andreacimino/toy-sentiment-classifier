
def evaluate(gold_file, result_file):
  # read gold standard and populate the count matrix
  gold = dict()
  gold_counts =  {'sub':{'0':0,'1':0},
                  'opos':{'0':0,'1':0},
                  'oneg':{'0':0,'1':0},
                  'iro':{'0':0,'1':0},
                  'lpos':{'0':0,'1':0},
                  'lneg':{'0':0,'1':0},
  }
  with open(gold_file) as f:
      for line in f:
          split_line =  line.rstrip().split(',')
          if "idtwitter" in split_line[0]:
              continue
          id, sub, opos, oneg, iro, lpos, lneg, top = map(lambda x: x, split_line[0:8])
          gold[id] = {'sub':sub, 'opos':opos, 'oneg':oneg, 'iro':iro, 'lpos': lpos, 'lneg': lneg}
          #gold_counts['sub'][sub]+=1
          gold_counts['opos'][opos]+=1
          gold_counts['oneg'][oneg]+=1
         # gold_counts['lpos'][lpos]+=1
        #  gold_counts['lneg'][lneg]+=1
        #  gold_counts['iro'][iro]+=1
  # read tweets IDs to exclude
  #if args.exclude_file:
   #   with open(args.exclude_file) as f:
   #       exclude = map(lambda x:x.rstrip(), f.readlines())
  #else:
  exclude = {}
  # read result data
  result = dict()
  with open(result_file) as f:
      for line in f:
          split_line =  line.rstrip().split(',')
          id, sub, opos, oneg, iro, lpos, lneg, top = map(lambda x: x, split_line[0:8])
  #        id = '"' + id + '"'
          if not id in exclude:
              result[id]= {'sub':sub, 'opos':opos, 'oneg':oneg, 'iro':iro, 'lpos':lpos,
                           'lneg': lneg}
  # evaluation: single classes
  for task in ['sub', 'opos', 'oneg', 'iro', 'lpos', 'lneg']:
      # table header
      print "\ntask: {}".format(task)
      print "prec. 0\trec. 0\tF-sc. 0\tprec. 1\trec. 1\tF-sc. 1\tF-sc."
      correct =  {'0':0,'1':0}
      assigned = {'0':0,'1':0}
      precision ={'0':0.0,'1':0.0}
      recall =   {'0':0.0,'1':0.0}
      fscore =   {'0':0.0,'1':0.0}

      # count the labels
      for id, gold_labels in gold.iteritems():
          #import pdb; pdb.set_trace()
          if not id in exclude: #ignore not unavailable tweets
           #   print "NOT EXCLUDE"
  #            print id in result
              if (not id in result) or result[id][task]=='':
                  pass
              else:
            #      print "ASSIGNING"
                  assigned[result[id][task]] += 1
                  if gold_labels[task]==result[id][task]:
                      correct[result[id][task]] += 1
      # compute precision, recall and F-score
      for label in ['0','1']:
          try:
              precision[label] = float(correct[label])/float(assigned[label])
              recall[label] = float(correct[label])/float(gold_counts[task][label])
              fscore[label] = (2.0 * precision[label] * recall[label]) / (precision[label] + recall[label])
          except:
              # if a team doesn't participate in a task it gets default 0 F-score
              fscore[label] = 0.0

      # write down the table
      print "{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}".format(
              precision['0'], recall['0'], fscore['0'],
              precision['1'], recall['1'], fscore['1'],
              (fscore['0'] + fscore['1'])/2.0)


  # polarity evaluation needs a further step
  print "\ntask: polarity"
  print "Combined F-score"
  correct =  {'opos':{'0':0,'1':0}, 'oneg':{'0':0,'1':0}}
  assigned = {'opos':{'0':0,'1':0}, 'oneg':{'0':0,'1':0}}
  precision ={'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}
  recall =   {'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}
  fscore =   {'opos':{'0':0.0,'1':0.0}, 'oneg':{'0':0.0,'1':0.0}}

  # count the labels
  for id, gold_labels in gold.iteritems():
      if not id in exclude: #ignore not unavailable tweets
          for cl in ['opos','oneg']:
              if (not id in result) or result[id][cl]=='':
                  pass
              else:
                  assigned[cl][result[id][cl]] += 1
                  if gold_labels[cl]==result[id][cl]:
                      correct[cl][result[id][cl]] += 1

  # compute precision, recall and F-score
  for cl in ['opos','oneg']:
      for label in ['0','1']:
          try:
              precision[cl][label] = float(correct[cl][label])/float(assigned[cl][label])
              recall[cl][label] = float(correct[cl][label])/float(gold_counts[cl][label])
              fscore[cl][label] = float(2.0 * precision[cl][label] * recall[cl][label]) / float(precision[cl][label] + recall[cl][label])
          except:
              fscore[cl][label] = 0.0

  fscore_opos = (fscore['opos']['0'] + fscore['opos']['1'] ) / 2.0
  fscore_oneg = (fscore['oneg']['0'] + fscore['oneg']['1'] ) / 2.0

  # write down the table
  print "{0:.4f}".format((fscore_opos + fscore_oneg)/2.0)



  # polarity evaluation needs a further step
  print "\ntask: lpolarity"
  print "Combined F-score"
  correct =  {'lpos':{'0':0,'1':0}, 'lneg':{'0':0,'1':0}}
  assigned = {'lpos':{'0':0,'1':0}, 'lneg':{'0':0,'1':0}}
  precision ={'lpos':{'0':0.0,'1':0.0}, 'lneg':{'0':0.0,'1':0.0}}
  recall =   {'lpos':{'0':0.0,'1':0.0}, 'lneg':{'0':0.0,'1':0.0}}
  fscore =   {'lpos':{'0':0.0,'1':0.0}, 'lneg':{'0':0.0,'1':0.0}}

  # count the labels
  for id, gold_labels in gold.iteritems():
      if not id in exclude: #ignore not unavailable tweets
          for cl in ['lpos','lneg']:
              if (not id in result) or result[id][cl]=='':
                  pass
              else:
                  assigned[cl][result[id][cl]] += 1
                  if gold_labels[cl]==result[id][cl]:
                      correct[cl][result[id][cl]] += 1

  # compute precision, recall and F-score
  for cl in ['lpos','lneg']:
      for label in ['0','1']:
          try:
              precision[cl][label] = float(correct[cl][label])/float(assigned[cl][label])
              recall[cl][label] = float(correct[cl][label])/float(gold_counts[cl][label])
              fscore[cl][label] = float(2.0 * precision[cl][label] * recall[cl][label]) / float(precision[cl][label] + recall[cl][label])
          except:
              fscore[cl][label] = 0.0

  fscore_lpos = (fscore['lpos']['0'] + fscore['lpos']['1'] ) / 2.0
  fscore_lneg = (fscore['lneg']['0'] + fscore['lneg']['1'] ) / 2.0

  # write down the table
  print "{0:.4f}".format((fscore_lpos + fscore_lneg)/2.0)
