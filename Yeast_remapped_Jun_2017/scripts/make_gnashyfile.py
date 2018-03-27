"""
make_gnashyfile.py
written 9/2/16 by RDM

updated 3/28/17 by RDM to account for yeast chromosome 
nomenclature and to give 1-indexed coordinates

last updated 4/3/17 to fix a problem with paired reads
and on 4/10/17 to require alignments to start at base 0 of 
hard clipped read
and again on 4/10/17 to count forward and reverse reads
that map to the same base as separate insertions.

tested 4/11/17 and fixed another bug.  Seems ok now.

usage:
make_gnashyfile -b <barcode filename>  -o <output path>

required
none

not required
-b <barcode filename>, default = ../raw/barcodes.txt
-p <path>, default = ../output_and_analysis/  path for input and output

"""

import argparse
import pysam
import pandas as pd
import csv
import sys

def sort_gnashy_file(gnashyfilename):
    gnashy_frame = pd.read_csv(gnashyfilename,delimiter='\t',header=None,names=['chr','pos','reads'])
    gnashy_frame = gnashy_frame.sort_values(['chr','pos'])
    gnashy_frame.to_csv(gnashyfilename,sep='\t',header=False,index=False)
    return [len(gnashy_frame),gnashy_frame.reads.sum()]



def read_barcode_file(barcode_filename):
#This function reads in the experiment name, the primer barcode, and the transposon barcodes
#from a file which is in the following format:
#expt name \t primer barcode \t transposon barcode 1,transposon barcode 2, transposon barcode 3 etc.
#It then returns a dictionary with key = a tuple (primer barcode, transposon barcode), value = expt_name
#The last line of the file should not have a return character
    reader = csv.reader(open(barcode_filename, 'r'),delimiter = '\t')
    d = {}
    for row in reader:
        exname,b1,b2 = row
        for transposon_barcode in b2.split(","):
            d[(b1,transposon_barcode)]=exname
    return d

def make_gnashyfile(bcfilename,outpath):
    #make chromosome list
    #mitrochondrial mappings break gnashy viewer.  Uncomment this when we switch over to the 
    #wustl browser.
    """chr_list = {"ref|NC_001133|","ref|NC_001134|","ref|NC_001135|", "ref|NC_001136|", \
            "ref|NC_001137|","ref|NC_001138|","ref|NC_001139|","ref|NC_001140|", \
            "ref|NC_001141|","ref|NC_001142|","ref|NC_001143|","ref|NC_001144|", \
            "ref|NC_001145|", "ref|NC_001146|", "ref|NC_001147|", "ref|NC_001148|","ref|NC_001224|"}
    chr_dict = {"ref|NC_001133|":1, "ref|NC_001134|":2, "ref|NC_001135|":3, "ref|NC_001136|":4, \
            "ref|NC_001137|":5, "ref|NC_001138|":6, "ref|NC_001139|":7, "ref|NC_001140|":8, \
            "ref|NC_001141|":9, "ref|NC_001142|":10, "ref|NC_001143|":11, "ref|NC_001144|":12, \
            "ref|NC_001145|":13, "ref|NC_001146|":14, "ref|NC_001147|":15, "ref|NC_001148|":16,"ref|NC_001224|:MT"}"""
    chr_list = {"ref|NC_001133|","ref|NC_001134|","ref|NC_001135|", "ref|NC_001136|", \
            "ref|NC_001137|","ref|NC_001138|","ref|NC_001139|","ref|NC_001140|", \
            "ref|NC_001141|","ref|NC_001142|","ref|NC_001143|","ref|NC_001144|", \
            "ref|NC_001145|", "ref|NC_001146|", "ref|NC_001147|", "ref|NC_001148|"}
    chr_dict = {"ref|NC_001133|":1, "ref|NC_001134|":2, "ref|NC_001135|":3, "ref|NC_001136|":4, \
            "ref|NC_001137|":5, "ref|NC_001138|":6, "ref|NC_001139|":7, "ref|NC_001140|":8, \
            "ref|NC_001141|":9, "ref|NC_001142|":10, "ref|NC_001143|":11, "ref|NC_001144|":12, \
            "ref|NC_001145|":13, "ref|NC_001146|":14, "ref|NC_001147|":15, "ref|NC_001148|":16}
    print "making yeast gnashyfile"
    #read in experiments and barcodes.  Key = (primer barcode, Transposon barode)
    #Value = expt name
    barcode_dict = read_barcode_file(bcfilename)
    #initialize quality control dictionary
    qc_dict = {}
    #LOOP THROUGH EXPERIMENTS
    #loop through experiments and make a separate gnashy file for each
    for expt in list(set(barcode_dict.values())):
      #for each experiment, there will be multiple bam files.  Loop through all of them
      #open output gnashyfile
      print "Analyzing "+expt
      output_filename = outpath+expt+".gnashy"
      output_handle = file(output_filename, 'w') 
      #LOOP THROUGH BAM FILES CORRESPONDING TO 1 experiment
      for key in barcode_dict.keys(): #this could be made more efficient, but its more clear this way
        if barcode_dict[key] == expt:
          primerBC = key[0]
          transposonBC = key[1]
          basename = outpath+expt+"_"+primerBC+"_"+transposonBC
          sbamFilename = basename+".sorted"
          pysam.sort(basename+".bam",sbamFilename)
          #sort and index bamfile
          sbamFilename = sbamFilename+".bam"
           
          pysam.index(sbamFilename)
          print sbamFilename
          #inialize gnashy dictionary
          for_gnashy_dict = {}
          rev_gnashy_dict = {}
          #make AlignmentFile object
          current_bamfile = pysam.AlignmentFile(sbamFilename,"rb")

          #loop through the chromosomes and pileup start sites
          for chr in chr_list:
            print chr
            aligned_reads_group = current_bamfile.fetch(chr)
            #now loop through each read and pile up start sites
            for aread in aligned_reads_group:
              if not(aread.is_read2):  #only count read1's
                #is the read a reverse read?
                if aread.is_reverse:
                    if aread.query_alignment_end == (aread.query_length): #read has to start right after primer
                      pos = aread.get_reference_positions()[-1]+1
                      if (chr,pos) in rev_gnashy_dict:
                          rev_gnashy_dict[(chr,pos)]+=1
                      else:
                          rev_gnashy_dict[(chr,pos)]=1
                elif aread.query_alignment_start == 0: #forward read has to start right after primer
                    pos = aread.get_reference_positions()[0]+1
                    if (chr,pos) in for_gnashy_dict:
                        for_gnashy_dict[(chr,pos)]+=1
                    else:
                        for_gnashy_dict[(chr,pos)]=1
          #output dictionary to gnashy file
          for key in for_gnashy_dict:
            output_handle.write("%s\t%s\t%s\n" %(chr_dict[key[0]],key[1], for_gnashy_dict[key] ))
          for key in rev_gnashy_dict:
            output_handle.write("%s\t%s\t%s\n" %(chr_dict[key[0]],key[1], rev_gnashy_dict[key] ))
      output_handle.close()
      #OPEN GNASHY FILE AND SORT BY CHR THEN POS
      qc_dict[expt] = sort_gnashy_file(output_filename)
    #after all experiments have been analyzed, print out qc
    qc_handle = file(outpath+"gnashyQC.txt",'w')
    for key in qc_dict:
      qc_handle.write("%s\t%s\t%s\n" %(key,qc_dict[key][0], qc_dict[key][1] ))
    qc_handle.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make_gnashyfile.py')
    parser.add_argument('-b','--barcodefile',help='barcode filename (full path)',required=False,default='../raw/barcodes.txt')
    parser.add_argument('-p','--outputpath',help='output path',required=False,default='../output_and_analysis')
    args = parser.parse_args()
    if not args.outputpath[-1] == "/":
        args.outputpath = args.outputpath+"/"
    make_gnashyfile(args.barcodefile,args.outputpath)


                            





