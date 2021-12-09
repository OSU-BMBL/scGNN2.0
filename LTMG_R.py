
import os
import numpy as np
import pandas as pd

import info_log

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

importr('scGNNLTMG')

def runLTMG(X, args):

    info_log.print('--------> Running LTMG ...')
    
    expression_file = os.path.join(args.output_dir, 'expression_for_LTMG.csv')
    output_file = os.path.join(args.output_dir, 'ltmg.txt')

    # np.savetxt(expression_file, X, delimiter=',')
    pd.DataFrame(X).to_csv(expression_file)

    # robjects.r('''
    #         setwd("/users/PAS1475/qiren081/GCNN/data/sc/ex")
    #         test.data <- read.csv("Biase_expression.csv",header = T,row.names = 1,check.names = F)
    #         object <- scGNNLTMG::CreateLTMGObject(as.matrix(test.data))
    #         object <- scGNNLTMG::RunLTMG(object,Gene_use = "all",seed =123,k=5)
    #         my.matrix <- cbind(ID = rownames(object@OrdinalMatrix),object@OrdinalMatrix)
    #         write.table(my.matrix, file = "LTMG_discretization_Bia.txt",row.names = F, quote = F,sep = "\t")
    # ''')
    robjects.globalenv['expressionFile'] = expression_file
    robjects.globalenv['output_file'] = output_file
    
    #Original version without sparse
    robjects.r('''           
        test.data <- read.csv(expressionFile, header = T, row.names = 1, check.names = F)
        object <- scGNNLTMG::CreateLTMGObject(as.matrix(test.data))
        object <- scGNNLTMG::RunLTMG(object, Gene_use = "all", seed = 123, k=5)
        my.matrix <- cbind(ID = rownames(object@OrdinalMatrix), object@OrdinalMatrix)
        write.table(my.matrix, file = output_file, row.names = F, quote = F, sep = "\t")
    ''')

#Sparse version
#R code:
# x <- read.csv("Use_expression_test.csv",header = T,row.names = 1,check.names = F)
# object <- CreateLTMGObject(x)
# object <-RunLTMG(object,Gene_use = 'all')
# WriteSparse(object,path='/storage/htc/joshilab/wangjue/')
#    robjects.r('''  
#        x <- read.csv(expressionFile,header = T,row.names = 1,check.names = F)
#        object <- CreateLTMGObject(x)
#        object <-RunLTMG(object,Gene_use = 'all')
#        WriteSparse(object,path=ltmgFolder,gene.name=FALSE, cell.name=FALSE)           
#    ''')