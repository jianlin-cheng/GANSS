
##################################  2017/11/03    ---- log by Jie

  	
  1. Organize the data for training based on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4348072/
  
  
  	* All data: 1425 proteins : /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/all_pdb.lst
  	* Train data: 1230 proteins : /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/dncon-train.lst
  	* Test data: 195 proteins : /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/adj_dncon-test.lst
  	* Independent evaluation data: 105 CASP9 : /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/casp9.lst
  	* Independent evaluation data: 93 CASP10 : /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/lists/casp10.lst
  	
  	* FASTA file:  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/fasta/
  	* PSSM file:  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/pssm/
  	* dssp file:  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/dssp/
  	* chains file:  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/chains/
  	* ssa file:  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/data/ssa/
  
  2. Dr.Cheng created the github, so clone it on lewis server 
  
  
  cd /storage/htc/bdm/jh7x3
  git clone https://github.com/jianlin-cheng/GANSS.git
  cd /storage/htc/bdm/jh7x3/GANSS
  
  ### working directory: /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss
  ### github directory: /storage/htc/bdm/jh7x3/GANSS
  
  	
  3. add 'generate_feature_for_convolution' into '/storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/lib/DN_SSpred.pm'
  
  	a. generate 1 window feature
  		$  cd  /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/scripts/
  	 	$  perl P1_generate_features.pl  -out  features_win1 -wind 1
  			
        ** output features: /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win1/
  			
  		
  		
  	b. generate 15 window feature
  		$  cd  /storage/htc/bdm/jh7x3/GANSS/Deep1Dconv_ss/scripts/
  	 	$  perl P1_generate_features.pl  -out  features_win1 -wind 15
  			
        ** output features:  /storage/htc/bdm/Collaboration/jh7x3/DeepCov_SS_SA_project/Deep1Dconv_ss/features_win15/
  			
  		
  4. todo:  test 1d convolutional neural network


#####################################
 
 
 
