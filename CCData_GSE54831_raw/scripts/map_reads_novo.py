"""
map_reads_novo.py
written 3/24/16
usage

map_reads -f <base file name> -g <genome> -p <paired end flag> 
-t transponson sequence to trim -l <length of sequence 3' to transposon
 to map> -q <quality filter> -o <output path>

required
-f <base file name> -g <genome> 

not required
-p <paired end flag> default = False 
-t transponson sequence to trim default = AATTCACTACGTCAACA
-l <length of sequence 3' to transposon to map> default = all
-q <quality filter>  default = 10
-o <output path> default = ../output_and_analysis

This program requires that the novoalign module is loaded and


It also requires the samtools module is loaded.
    """

import argparse
import os
import re
import sys

def map_reads(basefilename,genome,paired,transposon_sequence,map_len,qcutoff,outpath):
    outfilename = basefilename+".bam"
    outerrname = basefilename+".err"
    r1_filename = basefilename+".fastq"
    ##TODO: find out tag sequence, ignoring trimming tag seq for now
    # regex = r"([A,C,G,T]+)_[A,C,G,T]+$"
    # match = re.search(regex,basefilename)
    # barcode_sequence = match.group(1)
    barcode_sequence = ""
    trim_seq = barcode_sequence+transposon_sequence
    map_len = int(map_len) + len(trim_seq)
    print "filtering "+trim_seq
    
    if paired:
        r2_filename = basefilename+"_R2.fastq"
        novoalign_string = ("novoalign -o SAM -d "+genome+" -i PE 50-700 -f " +r1_filename+" "+r2_filename+" -5 "+trim_seq+" -n "+str(map_len)+" 2> "+outerrname+" |samtools view -bS -q "+str(qcutoff)+ 
        " > "+outfilename)    
    else:
        novoalign_string = ("novoalign -o SAM -d "+genome+" -f "+r1_filename+" -5 "+trim_seq+" -n "+str(map_len)+" 2> "+outerrname+" |samtools view -bS -q "+str(qcutoff)+" > "+outfilename)  
    print novoalign_string
    os.system(novoalign_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='map_reads.py')
    parser.add_argument('-f','--basename',help='base filename (full path)',required=True)
    parser.add_argument('-g','--genome',help='genome name',required=True)
    parser.add_argument('-p','--paired',help='paired read flag',required=False,default=False)
    parser.add_argument('-t','--transposon_sequence',help='transposon sequence',default="AATTCACTACGTCAACA")
    parser.add_argument('-l','--length',help='length of read to map',required=False,default=250)
    parser.add_argument('-q','--quality',help='quality score cutoff',required=False,default=10)
    parser.add_argument('-o','--outpath',help='output path',required=False,default='../output_and_analysis')
    args = parser.parse_args()
    if args.paired == "False":
        args.paired=False
    if not args.outpath[-1] == "/":
        args.outpath = args.outpath+"/"
    os.chdir(args.outpath)
    map_reads(args.basename,args.genome,args.paired,args.transposon_sequence,args.length,args.quality,args.outpath)

