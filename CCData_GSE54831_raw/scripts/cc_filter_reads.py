"""
cc_filter_reads.py
written 9/1/16 by RDM

Modified 4/3/16 to remove /n line from split files.

Read #1
The first 5 base pairs will be the primer barcode (index 0 to 4), followed by 17bp of transposon sequence (index 5 to 21):  AATTCACTACGTCAACA, after which is genomic sequence

Read #2
The first 8bp will be the transposon barcode, followed by the restriction enzyme bank and then genomic DNA.  So it will look like NNNNNNNN TCGA GCGC CCGG where the genomic sequence 
could start at any of the three restriction sites.

The barcode file should be of the following format:  #expt name \t primer barcode \t transposon barcode 1,transposon barcode 2, transposon barcode 3 etc.

usage 
python cc_filter_reads.py -r1 <read1 file> -r2 <read2 file> 
 -b<barcode file> -o <output path>
--hammp <hamming distance for primer barcode>
--hammt <hamming distance for transposon barcode>

required fields:
    -r1 read 1 filename (full path)
    -r2 read 2 filename (full path)

not required
    -b barcode file = ../raw/barcodes.txt
    -o output path = ../output_and_analysis
    -hp 0
    -tp 0
"""
import argparse
import csv
from Bio import SeqIO
from Bio import Seq
import os

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

def hamming_distance(s1, s2):
#Return the Hamming distance between equal-length sequences
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def filter_reads(read1fn,read2fn,barcodefn,outpath,hammp,hammt):
    #This function does the following:
    #1.  Reads barcodes and corresponding experiments into a dictionary
    #2.  Opens the read 1 and checks for transposon sequence
    #3.  If the tranposon sequence is present, it checks to see if the primer
    #barcode matches the transposon barcode
    #4.  If both filters are passed, it prints the reads to a file of the format:
    #exptname_primerbc_transposonbc_R1.fasta (and/or R2)
    #5.  It also prints a master file of all of the reads file
    #6.  The program then outputs a brief QC that lists the fraction of reads that have a 
    #transposon match, the fraction of reads that have matching primer and transposon barcodes
    #and the number of reads for each experiment and the total number of reads analyzed.

    #Define some infrequently changed variables

    FILTER_SEQUENCE = "AATTCACTACGTCAACA"
    PRIMER_BARCODE_START = 0  
    PRIMER_BARCODE_END = 4
    TRANSPOSON_BARCODE_START = 0
    TRANSPOSON_BARCODE_END = 7 
    #MATCH BARCODES TO EXPERIMENT
    barcode_dict = read_barcode_file(barcodefn)
    print "I have read in the experiment barcodes."
    
    #Put all filenames in this file for later
    filelist_filehandle = open(outpath+"cc_filelist.txt",'w')

    #Make  a dictionary of filehandles for each barcode pair and undetermined barcodes    
    r1_bcp_filehandle_dict = {} #dictionary that contains the handles for each barcode pair
    r2_bcp_filehandle_dict = {} #same as above, but for read 2

    for key in barcode_dict.keys():
        r1_filename = outpath+barcode_dict[key]+"_"+key[0]+"_"+key[1]+"_R1.fastq"
        #print the filename minus the _R1.fasta suffix 
        #This makes the next step easier
        print >> filelist_filehandle,outpath+barcode_dict[key]+"_"+key[0]+"_"+key[1]
        r1_bcp_filehandle_dict[key] = open(r1_filename,'w')
        if read2fn:
            r2_filename = outpath+barcode_dict[key]+"_"+key[0]+"_"+key[1]+"_R2.fastq"
            r2_bcp_filehandle_dict[key]=open(r2_filename,'w')
    filelist_filehandle.close()
    
    #Make a filehandle to dump undetermined reads
    r1Undet_filehandle = open(outpath+"undetermined_R1.fastq",'w')
    r2Undet_filehandle = open(outpath+"undetermined_R2.fastq",'w')
    
    #Make a filehandle for the QC information
    qc_filehandle = open(outpath+"qc_filter_reads.txt",'w')

    #get handles for read files
    r1Handle = open(read1fn,"rU") #open read1 file
    if read2fn:
        r2Handle = open(read2fn,"rU") #open read2 file
    
    #make iterators for index read and r2
    if read2fn:
        read2_record_iter = SeqIO.parse(r2Handle,"fastq")
    
    #initialize QC counters
    total_reads = 0
    reads_with_transposon_seq = 0
    matched_reads = 0
    expt_dict = {}
    for expt in barcode_dict.values():
        expt_dict[expt]=0

    #LOOP THROUGH READ 1 (and READ 2)
    for read1_record in SeqIO.parse(r1Handle,"fastq"):
        if read2fn:
            read2_record = next(read2_record_iter)
        total_reads += 1  # advance reads counter
        if FILTER_SEQUENCE in read1_record.seq[0:7+len(FILTER_SEQUENCE)]:
            reads_with_transposon_seq = reads_with_transposon_seq + 1
            primerbc = str(read1_record.seq[PRIMER_BARCODE_START:PRIMER_BARCODE_END+1]) 
            transbc = str(read2_record.seq[TRANSPOSON_BARCODE_START:TRANSPOSON_BARCODE_END+1])
            #try to correct barcodes if the hamming distance cutoff is g.t. zero
            if ((hammp > 0) or (hammt > 0)):
                for key in barcode_dict.keys():
                    if hamming_distance(primerbc,key[0])<=hammp:
                        if hamming_distance(transbc,key[1])<= hammt:
                            primerbc = key[0]
                            transbc = key[1]
            #print primerbc,transbc
            #is primer_bc transposon barcode pair in dictionary?   
            if (primerbc,transbc) in barcode_dict:
                #if so, increment matched reads
                matched_reads += 1
                #update reads for the experiment
                expt_name = barcode_dict[(primerbc,transbc)]
                expt_dict[expt_name] += 1
                #output reads to the correct place
                print >> r1_bcp_filehandle_dict[(primerbc,transbc)],read1_record.format("fastq")[0:len(read1_record.format("fastq"))-1]
                if read2fn:
                    print >> r2_bcp_filehandle_dict[(primerbc,transbc)],read2_record.format("fastq")[0:len(read2_record.format("fastq"))-1]
            else: #if there is no match, print reads to undetermined file
                print >> r1Undet_filehandle,read1_record.format("fastq")[0:len(read1_record.format("fastq"))-1]
                print >> r2Undet_filehandle,read2_record.format("fastq")[0:len(read2_record.format("fastq"))-1]
        else: #if there is no match, print reads to undetermined file
            print >> r1Undet_filehandle,read1_record.format("fastq")[0:len(read1_record.format("fastq"))-1]
            print >> r2Undet_filehandle,read2_record.format("fastq")[0:len(read2_record.format("fastq"))-1]
    #print QC values to file
    
    
    print >> qc_filehandle,"There were "+str(total_reads)+" total reads"
    print >> qc_filehandle,str(reads_with_transposon_seq/float(total_reads))+" of the reads had a transposon sequence."
    print >> qc_filehandle,str(matched_reads/float(total_reads))+" of the total reads also had matched barcodes."
    for key in expt_dict.keys():
        print >> qc_filehandle, str(key)+"\t"+str(expt_dict[key])
    qc_filehandle.close()

    #Close all filehandles
    r1Handle.close()
    r1Undet_filehandle.close()
    for key in r1_bcp_filehandle_dict.keys():
        r1_bcp_filehandle_dict[key].close()
    if read2fn:
        r2Handle.close()
        r2Undet_filehandle.close()
        for key in r2_bcp_filehandle_dict.keys():
            r2_bcp_filehandle_dict[key].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cc_filter_reads.py')
    parser.add_argument('-r1','--read1',help='Read 1 filename (full path)',required=True)
    parser.add_argument('-r2','--read2',help='Read2 filename (full path)',required=True)
    parser.add_argument('-o','--outputpath',help='output path',required=False,default='../output_and_analysis/')
    parser.add_argument('-b','--barcodefile',help='barcode filename (full path)',required=False,default='../raw/barcodes.txt')
    parser.add_argument('--hammp',help='Primer barcode hamming distance',required=False,default=0)
    parser.add_argument('--hammt',help='Transposon barcode hamming distance',required=False,default=0)
    args = parser.parse_args()
    if not args.outputpath[-1] == "/":
        args.outputpath = args.outputpath+"/"
    filter_reads(args.read1,args.read2,args.barcodefile,args.outputpath,int(args.hammp),int(args.hammt))

    

