
import os
import numpy as np
import pandas as pd

import info_log

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

importr('scGNNLTMG')

def runLTMG(X, args):

    info_log.print('--------> Running LTMG ...')
    
    output_dir = os.path.join(args.output_dir, 'preprocessed_data')
    expr_file_name = "dropout_top_expression.csv" if args.dropout_prob else 'original_top_expression.csv'
    expression_file = os.path.join(output_dir, expr_file_name)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        pd.DataFrame(data=X['expr_b4_log'].T, index=X['gene'], columns=X['cell']).to_csv(expression_file)

    output_file = os.path.join(output_dir, f'LTMG_{args.dropout_prob}.csv')

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
        x <- read.csv(expressionFile, header = T, row.names = 1, check.names = F)
        object <- scGNNLTMG::CreateLTMGObject(x)
        object <- scGNNLTMG::RunLTMG(object, Gene_use = "all")
        my.matrix <- cbind(ID = rownames(object@OrdinalMatrix), object@OrdinalMatrix)
        write.table(my.matrix, file = output_file, row.names = F, quote = F, sep = "\t")
    ''') # output gene * cell

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
