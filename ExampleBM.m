%A sample learning of a bOltzmann machine representing Adder circuit using
%GenDisBM class
%To access DWave functions, add following folder to path
%/ sapi-matlab-client-3.0-win64 
clc
clear
close all

%===============================Initialize BM=================================

NumVisibleNode=7;   %Number of Visible nodes
NumOutputNode=3;    %Number of Output nodes 
                    %Should be strictly less than NumVisibleNode
NumHiddenNode=2;    %Number of Hidden nodes

Connection = triu(ones(NumVisibleNode+NumHiddenNode)); %Adjacency matric for Fully connected graph

%Initialize using constructor
testBM =  GenDisBM('bm', NumVisibleNode,NumOutputNode, NumHiddenNode, Connection);


%=================================Input Data==================================
%Define data as a matrix: [NumVisibleNodes x NumDataStates]
        
%Each column represents a visible data in the format [v_i;v_o]
%Each v_i should have a unique v_o
%        D1 D2 D3 D4 ................................D16
data =   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1;  %v_i1
          0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1;  %v_i2
          0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1;  %v_i3
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;  %v_i4
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1;  %v_o1
          0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1;  %v_o1
          0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]; %v_o2

%Insert data      
testBM.InputData(data,[])

%=================================Define cost=================================
%Inverse temperature.
Beta = 10;      % Not used for learning algorithms using DWave. Should be positive
%Cost = Alpha*KL_Divergence + (1-Alpha)*NegCondition-Log-Likelihood
Alpha = 0.5;    % Should be between 0 and 1
testBM.SetCostParameters(Alpha,Beta);

%=============================Define constraints==============================
%This is optional, if no constraints are required this section can be
%commented
%Parameters are indexed as 
%[H_1, ..., H_n, J_1, ..., J_m]
%The indexes of J can be found using testBM.EdgeMapInverse

%Matrix with #Row = NumConstrain #Column = NumParam
Acons = [ones(1,testBM.NumFieldParam),zeros(1,testBM.NumInteractionParam); ...
        zeros(1,testBM.NumFieldParam), ones(1,testBM.NumInteractionParam);]; 

%Vector with #Row = NumConstrain #Column = 1
bcons = [0;
         0];          
testBM.ApplyConstraints(Acons,bcons);

%===========Define Learning parameters for Momentum based method==============

%For First-order learning
Learning_rate = 0.1;
Weight_decay_rate = 0.0001;
Momentum_rate = 0.00;

testBM.SetLearningParameters(Learning_rate,Weight_decay_rate,Momentum_rate)

 
%===================Directly updating energy paramters =======================

%Optional (can be used for using helper functions)
testFieldParam = 2*testBM.MaxField *(rand(testBM.NumFieldParam,1)-0.5);
testInteractionParam = 2*testBM.MaxInteraction.*(rand(testBM.NumInteractionParam,1)-0.5);
testBM.UpdateGraphStrength(testFieldParam,testInteractionParam);



%=========================== Learning Step ===================================
%Set testBM.DWaveLearnFlag to 1 for DWave based learning
%                             0 for enumeration based learning
testBM.DWaveLearnFlag = 1;

%For first order momentum rate learning
testBM.FirstOrderFlag=1;
LearningSteps=100; %Number of Maximum Learning steps 
StartParameter = []; %To start from a random paramter

%Run optimization
testBM.OptimizeBM(LearningSteps, StartParameter)

%For second order learning
testBM.FirstOrderFlag=0;
LearningSteps=50; %Number of Maximum Learning steps 
StartParameter = testBM.CurrPoint; %To start from current paramters

%Run optimization
testBM.OptimizeBM(LearningSteps, StartParameter)

%===================Plotting and accessing parameters=========================
%Plotting command - works better for sparse graphs
testBM.PlotGraph
%Accessing energies
H = testBM.FieldStrength;
J = testBM.FieldStrength;

%========================== Comparison with DWave ============================
%To be used when both DWave and exact enumeration technique can be used
testBM.CompareDWave

%===== Plot KLDivergence and Negative Conditional Log-Likelihood wrt beta=====
%To be used when both DWave and exact enumeration technique can be used
BetaVals = logspace(-2,2,30);
[BetaVals, KLDivVals, NCLLVals] = testBM.PlotBMBetaCharacterisctic(BetaVals);