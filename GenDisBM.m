%GenDisBM - GENerative DIScriminative Boltzmann Machine
%This class offers functionality to learn a Boltzmann machine using direct
%enumeration techniques and DWave machine
%Created by Siddhartha Srivastava (sidsriva[at]umich.edu)
%Last update 8/24/2020
classdef GenDisBM < handle
    properties
        Name %Name of the BM
        NumVisibleNode %Number Visible node
        NumOutputNode %Number Output node 
        NumHiddenNode %Number of Hidden nodes 
        NumInputNode %Number Input node 
        NumAllNode %Number of All node
        %Note If using DWave following values wont work
        %NumHiddenNode = 1
        %NumOutputNode+NumHiddenNode = 1 
        
        VisibleDataStates %Visible Data States = [InputData;OutputData]
        VisibleDataPMF %Probability mass function of Visible data
        NumVisibleDataStates % Number of Visible data states
        InputDataStates %Input Data States
        InputDataPMF %Probability mass function of Input data
        NumInputDataStates % Number of Input data states
        NumAllStates % Number of Total states = 2^NumAllNodes
        
        Connection %Upper triangle matrix of the adjacency
        NumFieldParam % Number of field terms same as number of vertices
        NumInteractionParam % Number of interaction terms same as number of edges
        NumParam % Number of total parameters = NumFieldParam + NumInteractionParam
        EdgeMap % Edge number to the index of adjacency matrix
        EdgeMapInverse % Matrix reads the value of index of edge at the respective matrix element
        
        MaxField=1 %scalar - Maximum field strength absolute
        MaxInteraction=1 %scalar - Maximum interaction strength absolute
        
        %Linear Constrain of Format ALinCon*x - bLinCon = 0
        NumConstraint=0 %Number of additional linear constrains
        ALinCon=[] %Matrix with #Row = NumConstrain #Column = NumParam
        bLinCon=[] %Vector with #Row = NumConstrain #Column = 1
        
        %DWave Structures
        DWGraph %DWave Structure for complete graph
        DWHiddenGraph %DWave Structure for hidden graph
        DWOutputHiddenGraph %DWave Structure for hidden graph
        NumreadFactor = 50 % # of DWave reads, prescribed value 50 
        
        %BM Properties
        FieldStrength %H value for each vertex
        InteractionStrength %Interaction strength for all the edges (indexed using edge map)
        KLDiv %KLDivergence of current BM
        NCondLogLike %Negative Conditional Log-Likelihood
        
        %Flags
        EstimateHessianFlag = 1 %To Estimate hessian while calculating gradients.
        %Turned on when using fmincon and off when using 1st order method
        FirstOrderFlag = 0      %To learn via first order schemes
        DWaveLearnFlag = 0      %To learn from DWave
        
        %Current cost values and parameter evaluated during optimization
        CurrPoint %Last estimated parameter array
        CostVal %Last estimated cost
        GradientVal %Last estimated gradient
        HessianVal %Last estimated hessian
        
        %Matrices to be used during ExactBoltzEstimate
        States = [] % NumVertex x NumStates matrix for enumerating all states
        Energy_grad = [] % NumStates x NumParam matrix representing del E/ del theta
        VisibleStartIndex =[] % NumVisibleStates x 1 Index for first state corresponding to each visible state
        InputStartIndex =[] % NumVisibleStates x 1 Index for first state corresponding to each input state
        
        %This is where you update the parameters for learning parameters
        %for gradient learning.
        Learning_parameters
        % It also contains the momentum information
        % while in learning loop
        % Structure contains following variables
        %Learning_parameters.Learning_rate: Learning rate for gradient scheme;
        %Learning_parameters.Weight_decay_rate: Weight decay rate for gradient scheme;
        %Learning_parameters.Momentum_rate: Momentum_rate for gradient scheme;
        %Exhaustive search, default value is 1.
        
        CostParameters %This is where you update the parameters for learning
        %CostParameters.Beta: To be used for setting temperature (Only used in Exhaustive search)
        %CostParameters.Alpha: Cost = Alpha*KL_Divergence + (1-Alpha)*NegCondition-Log-Likelihood
    end
    methods
        function obj = GenDisBM(Name, NumVisibleNode,NumOutputNode, NumHiddenNode, Connection)
            %Constructor for the class BoltzmannMachine
            obj.Name = Name;
            obj.NumVisibleNode = NumVisibleNode;
            obj.NumOutputNode = NumOutputNode;
            obj.NumHiddenNode = NumHiddenNode;
            obj.NumInputNode = NumVisibleNode - NumOutputNode;
            obj.NumAllNode = NumVisibleNode + NumHiddenNode;
            obj.NumAllStates = 2^obj.NumAllNode;
            assert(size(Connection,1)== NumVisibleNode+ NumHiddenNode && ...
                size(Connection,2)== NumVisibleNode+ NumHiddenNode,'Wrong size of Connection matrix');
            obj.Connection = heaviside(triu(Connection,1))*2-1;
            obj.NumFieldParam = NumVisibleNode+ NumHiddenNode;
            obj.NumInteractionParam = sum(sum(obj.Connection));
            obj.NumParam = obj.NumFieldParam + obj.NumInteractionParam;
            %Define Edge maps
            obj.EdgeMapInverse = obj.Connection;
            obj.EdgeMap = zeros(obj.NumInteractionParam,2);
            temp = find(obj.Connection==1);
            for i=1:length(temp)
                tempx = 1 + mod(temp(i)-1 ,NumVisibleNode+ NumHiddenNode);
                tempy = 1+ ((temp(i)-tempx) / (NumVisibleNode+ NumHiddenNode));
                obj.EdgeMap(i,:) = [tempx, tempy];
                obj.EdgeMapInverse(tempx, tempy) = i;
            end
            %Set default Cost paramters
            Alpha = 0.5; Beta = 1;
            obj.SetCostParameters(Alpha,Beta)
        end
        function InputData(obj,VisibleDataStates,VisibleDataPMF)
            %Function to Input data states
            %VisibleDataStates: Matrix with NumRows = Visible Nodes,
            %                               NumColumn: Number of data points
            %VisibleDataPMF: Row vector with probability of each data
            %point; Could be left as []
            assert(size(VisibleDataStates,1) == obj.NumVisibleNode,'Number of Rows in data set should be equal to the visible nodes');
            if isempty(VisibleDataPMF)
                VisibleDataPMF = ones(1,size(VisibleDataStates,2))./ size(VisibleDataStates,2);
            end
            assert(size(VisibleDataStates,2) == length(VisibleDataPMF),'Number of columns in data state do not match the number of PMF');
            assert(prod(VisibleDataPMF)>0 ,'All probabilities should be positive');
            assert(abs(sum(VisibleDataPMF)-1)<eps ,'All probabilities should sum to 1');
            
            obj.NumVisibleDataStates = length(obj.VisibleDataPMF);
            [temp_C,temp_ia,~] = unique(VisibleDataStates(1:obj.NumInputNode,:)'...
                ,'rows');
            obj.VisibleDataStates = VisibleDataStates(:,temp_ia);
            obj.VisibleDataPMF = VisibleDataPMF(temp_ia);
            obj.NumVisibleDataStates = length(obj.VisibleDataPMF);
            obj.InputDataStates = temp_C';
            obj.InputDataPMF = VisibleDataPMF(temp_ia);
            obj.NumInputDataStates = length(obj.InputDataPMF);
            assert(obj.NumInputDataStates==obj.NumVisibleDataStates,...
                'Function approximation invalid: non-unique output to input');
            %Set default Learning parameters for 1st order Learning
            Learning_rate = 0.1;
            Weight_decay_rate = 0.0001;
            Momentum_rate = 0.0;
            obj.SetLearningParameters(Learning_rate,Weight_decay_rate,Momentum_rate);
        end
        
        function SetLearningParameters(obj,Learning_rate,Weight_decay_rate,Momentum_rate)
            %Function to set Learning parameters for 1st order learning
            %Input: 
            %Theta(t+1) = Theta(t) + D_Theta(t)
            %D_Theta(t) = - Learning_rate * dI/dTheta ...
            %             - Weight_decay_rate * Theta(t) ...
            %             + Momentum_rate * D_Theta(t-1)
            %Approximate values:
            %Learning_rate ~ Undetermined
            %Weight_decay_rate ~ 0.01-0.00001
            %Momentum_rate ~ 0.9 (initially 0.5)
            obj.Learning_parameters.Learning_rate = Learning_rate;
            obj.Learning_parameters.Weight_decay_rate = Weight_decay_rate;
            obj.Learning_parameters.Momentum_rate = Momentum_rate;
        end
        
        function SetCostParameters(obj,Alpha,Beta)
            %Function to set Cost parameters
            %Input: 
            %Beta       To be used for setting temperature (Only used in Exhaustive search)
            %Alpha      Cost = Alpha*KL_Divergence + (1-Alpha)*NegCondition-Log-Likelihood
            assert(Alpha>=0 && Alpha<=1,'Alpha must be between 0 and 1')
            assert(Beta>0,'Beta must be strictly positive')
            obj.CostParameters.Beta = Beta;
            obj.CostParameters.Alpha = Alpha;
        end        
        
        function ApplyConstraints(obj,Aeq,beq)
            %Function to set equality constraints for parameters
            %Only to be used with second order learning
            %Input: 
            %Aeq       Matrix with #Row = NumConstrain #Column = NumParam
            %beq       Vector with #Row = NumConstrain #Column = 1       
            assert(size(Aeq,1)==size(beq,1),'Number of constraints do not match');
            assert(size(Aeq,2)==obj.NumFieldParam+obj.NumInteractionParam || size(Aeq,2)==0 ...
                ,'Number of columns do not match the number of Parameters');
            assert(size(beq,2)==1 || size(beq,2)==0,'b should be a vector');
            obj.NumConstraint=size(beq,1);
            obj.ALinCon = Aeq;
            obj.bLinCon = beq;
            obj.Learning_parameters.Momentum.Lambda = zeros(obj.NumConstraint,1);
        end
        function CreateDWaveStructure(obj)
            %Function to initialize the DWave structures
            %Complete graph
            obj.DWGraph = DWaveStructure(strcat('Graph_',obj.Name), obj.Connection, 1);
            %Hidden graph - To marginalize over visible states
            obj.DWHiddenGraph = DWaveStructure(strcat('Graph_hidden_',obj.Name),...
                obj.Connection(obj.NumVisibleNode+1 :end , obj.NumVisibleNode +1:end), 1);
            %Output-Hidden graph - To marginalize over Input data states
            obj.DWOutputHiddenGraph = DWaveStructure(strcat('Graph_outputhidden_',obj.Name),...
                obj.Connection(obj.NumInputNode+1 :end , obj.NumInputNode+1:end), 1);
            
        end
        function UpdateGraphStrength(obj,H,J)
            %Function to update the Graph Strength (Works in 0/1 format)
            assert(length(H) == obj.NumFieldParam,'Number of field paramters do no match total number of nodes');
            obj.FieldStrength = H(:);
            if size(J,1)==(obj.NumVisibleNode+ obj.NumHiddenNode) && size(J,2)==(obj.NumVisibleNode+ obj.NumHiddenNode)
                obj.InteractionStrength = J;
            elseif (size(J,1)==obj.NumInteractionParam && size(J,2)==1) || (size(J,2)==obj.NumInteractionParam && size(J,1)==1)
                obj.InteractionStrength = zeros(obj.NumHiddenNode+obj.NumVisibleNode);
                for i=1:obj.NumInteractionParam
                    obj.InteractionStrength(obj.EdgeMap(i,1),obj.EdgeMap(i,2)) = J(i);
                end
            else
                error('Size of J needs to be either same as adjacency(upper triangle) or [numedges,1]');
            end
            
        end
        
        function OptimizeBM(obj,MaxSteps,StartPoint)
            %Function to start learning process
            %Input: 
            % MaxSteps: Maximum number of steps tp be allowed by any
            %           algorithm
            % StartPoint: [1 x NumParam] - a starting value for gradient
            %             based method.
            %               - set to [] for a random start point
            %               - set to obj.CurrPoint to start from current
            
            if size(StartPoint,2)~=obj.NumParam || size(StartPoint,1)~=1
                display('Invalid starting point - Starting with a random value')
                StartPoint = rand(1,obj.NumParam);
            end
            obj.CurrPoint = StartPoint;
            
            %Setup Common data
            if obj.DWaveLearnFlag
                %Create DWave structure
                obj.CreateDWaveStructure()
                %Higher Numreads = Higher accuracy
                obj.DWGraph.Numreads = obj.NumreadFactor*obj.NumVisibleDataStates; 
                obj.DWHiddenGraph.Numreads = obj.NumreadFactor;
                obj.DWOutputHiddenGraph.Numreads = obj.NumreadFactor;
            else
                %Create eneumerated States and Energy_grad = (del E)/(del theta)
                SetIndex = 0:obj.NumAllStates-1; %Enumerate all states
                obj.States = ((double(dec2bin(SetIndex'))-48)*2-1)';
                obj.States = (obj.States+1)/2;
                obj.Energy_grad = zeros(length(SetIndex),obj.NumParam);
                obj.Energy_grad(:,1:obj.NumFieldParam) = obj.States';
                for j=1:obj.NumInteractionParam
                    obj.Energy_grad(:,obj.NumFieldParam+j) = ...
                        (obj.States(obj.EdgeMap(j,1),:).*obj.States(obj.EdgeMap(j,2),:))';
                end
                %Estimate InputStartIndex
                temp = num2str([obj.InputDataStates',...
                    zeros(obj.NumInputDataStates,obj.NumOutputNode + obj.NumHiddenNode)]);
                obj.InputStartIndex = bin2dec(temp)+1;
                %Estimate VisibleStartIndex
                temp = num2str([obj.VisibleDataStates',...
                    zeros(obj.NumVisibleDataStates,obj.NumHiddenNode)]);
                obj.VisibleStartIndex = bin2dec(temp)+1;
            end
            
            if obj.FirstOrderFlag
                %Linear constraints do not work in this method
                obj.EstimateHessianFlag = 0; %No Hessian needed in this method
                obj.Learning_parameters.Momentum = zeros(size(obj.CurrPoint)); %Initialize momentum
                obj.Learning_parameters.MinMomentum = 1e-10; %Cut-off criterion
                
                for i=1: MaxSteps
                    if obj.DWaveLearnFlag
                        [func,grad] = obj.DWaveBoltzEstimate(obj.CurrPoint,[]);
                        PrintSteps = 20;
                    else
                        [func,grad] = obj.ExactBoltzEstimate(obj.CurrPoint,[]);
                        PrintSteps = 100;
                    end
                    %Update step
                    delx = -obj.Learning_parameters.Learning_rate*grad'  ...
                        - obj.Learning_parameters.Weight_decay_rate * obj.CurrPoint  ...
                        + obj.Learning_parameters.Momentum_rate * obj.Learning_parameters.Momentum;
                    
                    x_new = obj.CurrPoint + delx;
                    
                    temp1 = max(abs(x_new(1:obj.NumFieldParam)));
                    temp2 = max(abs(x_new(obj.NumFieldParam+1:obj.NumParam)));
                    %Check if New strengths are higher than the bounds. If yes then decrease the
                    %learning rate to stay on the boundary
                    if temp1>=obj.MaxField || temp2>=obj.MaxInteraction
                        delta = max(temp1/obj.MaxField,temp2/obj.MaxInteraction) + eps;
                        x_new = x_new./delta;
                        delx = x_new - obj.CurrPoint;
                    end
                    %Update Value
                    obj.Learning_parameters.Momentum = delx;
                    obj.CurrPoint = x_new;
                    momentum_norm = max(abs(obj.Learning_parameters.Momentum));
                    if rem(i,PrintSteps) == 1
                        fprintf('Step: %i,\t KLDiv: %f\t NCLL: %f, \t Cost: %f, \t Learning Momentum: %f\n'...
                            ,i,obj.KLDiv,obj.NCondLogLike,func,momentum_norm);
                    end
                    if momentum_norm < obj.Learning_parameters.MinMomentum
                        break;
                    end
                end
                
            else
                obj.EstimateHessianFlag = 1;
                options = optimoptions('fmincon','Algorithm','interior-point',...
                    'SpecifyConstraintGradient',true,'SpecifyObjectiveGradient',true,...
                    'HessianFcn',@obj.getHessian,...
                    'MaxIterations',MaxSteps,...
                    'OutputFcn',{@obj.fminconRunTimeOutput});
                options.StepTolerance = 1.000000e-13;
                
                A = [];
                b = [];
                Aeq = obj.ALinCon;
                beq = obj.bLinCon;
                lb = -[obj.MaxField * ones(obj.NumFieldParam,1); ...
                    obj.MaxInteraction*ones(obj.NumInteractionParam,1)];
                ub = [obj.MaxField * ones(obj.NumFieldParam,1); ...
                    obj.MaxInteraction*ones(obj.NumInteractionParam,1)];
                x0 = StartPoint(:);
                NonLinearConstraint = [];
                if obj.DWaveLearnFlag
                    %Run optimization
                    [x,fval,eflag,output] = fmincon(@obj.DWaveBoltzEstimate,x0,...
                        A,b,Aeq,beq,lb,ub,NonLinearConstraint,options);
                else
                    %Run optimization
                    [x,fval,eflag,output] = fmincon(@obj.ExactBoltzEstimate,x0,...
                        A,b,Aeq,beq,lb,ub,NonLinearConstraint,options);
                end
                obj.UpdateGraphStrength(x(1:obj.NumFieldParam),x(obj.NumFieldParam+1:end));
                disp(fval)
                disp(eflag)
                disp([output.funcCount,output.iterations])
                obj.CurrPoint = x(:)';
            end
        end
        
        function [func,grad] = ExactBoltzEstimate(obj,x,~)
            %Function to estimate cost values and derivatives during
            %learning using Exact enumeration of states - should be used on small graphs       
            obj.UpdateGraphStrength(x(1:obj.NumFieldParam),x(obj.NumFieldParam+1:end));
            
            %Define Important Local Variables
            Pr_v = zeros(obj.NumVisibleDataStates,1); %p(.|v)
            Pr_i = zeros(obj.NumInputDataStates,1); %p(.|v_i)
            Ex_dEdth_v = zeros(1,obj.NumParam,obj.NumVisibleDataStates); %E_v(dE/dth)
            Ex_dEdth_i = zeros(1,obj.NumParam,obj.NumInputDataStates); %E_{v_i}(dE/dth)
            if obj.EstimateHessianFlag
                Ex_dEdth2 = zeros(obj.NumParam,obj.NumParam); %E(dE/dth_i dE/dth_j)
                Ex_dEdth2_v = zeros(obj.NumParam,obj.NumParam,obj.NumVisibleDataStates); %E_v(dE/dth_i dE/dth_j)
                Ex_dEdth2_i = zeros(obj.NumParam,obj.NumParam,obj.NumInputDataStates); %E_{v_i}(dE/dth_i dE/dth_j)
                Cov_dEdth_v = zeros(obj.NumParam,obj.NumParam,obj.NumVisibleDataStates); %Cov_{v}(dE/dth)
                Cov_dEdth_i = zeros(obj.NumParam,obj.NumParam,obj.NumInputDataStates); %Cov_{v_i}(dE/dth)
            end
            %=============================Estimate Energy=============================
            
            Energy = zeros(obj.NumAllStates,1);
            for i=1:obj.NumAllStates
                Energy(i) = obj.FieldStrength'*obj.States(:,i)  +...
                    (obj.States(:,i)'*obj.InteractionStrength*obj.States(:,i));
            end
            
            %======================Estimate Probability========================
            Z = sum(exp(-Energy*obj.CostParameters.Beta));
            Pr = exp(-Energy*obj.CostParameters.Beta)./Z;
            
            
            %======================Estimate Expectations========================
            
            %Without conditioning
            
            Ex_dEdth = Pr'*obj.Energy_grad;
            
            if obj.EstimateHessianFlag
                for i=1:obj.NumParam
                    for j=1:obj.NumParam
                        Ex_dEdth2(i,j) =  Pr'* (obj.Energy_grad(:,i) .*obj.Energy_grad(:,j));
                    end
                end
                Cov_dEdth = Ex_dEdth2 - Ex_dEdth'*Ex_dEdth;
            end
            
            %Conditioning on Visible data
            VisibleSize = 2^(obj.NumHiddenNode);
            for k =1:obj.NumVisibleDataStates
                state_subIndex = (obj.VisibleStartIndex(k): ...
                    obj.VisibleStartIndex(k) + VisibleSize-1)' ;
                Pr_v(k) = sum(Pr(state_subIndex));
                state_subPr = Pr(state_subIndex)./Pr_v(k);
                Ex_dEdth_v(:,:,k) = state_subPr'*obj.Energy_grad(state_subIndex,:);
                
                if obj.EstimateHessianFlag
                    for i=1:obj.NumParam
                        for j=1:obj.NumParam
                            Ex_dEdth2_v(i,j,k) =  state_subPr'* ...
                                (obj.Energy_grad(state_subIndex,i) .*obj.Energy_grad(state_subIndex,j));
                        end
                    end
                    Cov_dEdth_v(:,:,k) = Ex_dEdth2_v(:,:,k) - Ex_dEdth_v(:,:,k)'*Ex_dEdth_v(:,:,k);
                end
            end
            
            %Conditioning on Input data
            InputSize = 2^(obj.NumOutputNode + obj.NumHiddenNode);
            for k =1:obj.NumInputDataStates
                state_subIndex = (obj.InputStartIndex(k): ...
                    obj.InputStartIndex(k) + InputSize-1)' ;
                Pr_i(k) = sum(Pr(state_subIndex));
                state_subPr = Pr(state_subIndex)./Pr_i(k);
                Ex_dEdth_i(:,:,k) = state_subPr'*obj.Energy_grad(state_subIndex,:);
                if obj.EstimateHessianFlag
                    for i=1:obj.NumParam
                        for j=1:obj.NumParam
                            Ex_dEdth2_i(i,j,k) =  state_subPr'* ...
                                (obj.Energy_grad(state_subIndex,i) .*obj.Energy_grad(state_subIndex,j));
                        end
                    end
                    Cov_dEdth_i(:,:,k) = Ex_dEdth2_i(:,:,k) - Ex_dEdth_i(:,:,k)'*Ex_dEdth_i(:,:,k);
                end
            end
            
            %======================Estimate Gradient and Hessian========================
            grad = -obj.CostParameters.Alpha * Ex_dEdth;
            for k=1:obj.NumVisibleDataStates
                const = (obj.CostParameters.Alpha * obj.VisibleDataPMF(k) )+ ...
                    ((1-obj.CostParameters.Alpha)./obj.NumVisibleDataStates);
                grad = grad + const*Ex_dEdth_v(:,:,k);
            end
            for k=1:obj.NumInputDataStates
                const = (1-obj.CostParameters.Alpha)./obj.NumInputDataStates;
                grad = grad - const*Ex_dEdth_i(:,:,k);
            end
            if obj.EstimateHessianFlag
                hess = obj.CostParameters.Alpha * Cov_dEdth;
                for k=1:obj.NumVisibleDataStates
                    const = (obj.CostParameters.Alpha * obj.VisibleDataPMF(k) )+ ...
                        ((1-obj.CostParameters.Alpha)./obj.NumVisibleDataStates);
                    hess = hess - const*Cov_dEdth_v(:,:,k);
                end
                for k=1:obj.NumInputDataStates
                    const = (1-obj.CostParameters.Alpha)./obj.NumInputDataStates;
                    hess = hess + const*Cov_dEdth_i(:,:,k);
                end
            else
                hess=0;
            end
            %======================Estimate function value ========================
            obj.KLDiv = sum(obj.VisibleDataPMF.*log(obj.VisibleDataPMF./Pr_v'));
            obj.NCondLogLike = -sum(log(Pr_v./Pr_i));
            func = obj.CostParameters.Alpha * obj.KLDiv + ...
                ((1-obj.CostParameters.Alpha)./obj.NumInputDataStates)*obj.NCondLogLike;
            
            %======================Update Cost values ========================
            grad=grad';
            obj.CostVal = func;
            obj.GradientVal = grad;
            obj.HessianVal = hess;
        end
        
        function [func,grad] = DWaveBoltzEstimate(obj,x,~)
            %Function to estimate cost values and derivatives during
            %learning using DWave
            obj.UpdateGraphStrength(x(1:obj.NumFieldParam),x(obj.NumFieldParam+1:end));
            
            %Define Important Local Variables
            Pr_v_i = zeros(obj.NumVisibleDataStates,1); %p(.|v)
            Pr_v = zeros(obj.NumVisibleDataStates,1); %p(.|v_i)
            Ex_dEdth_v = zeros(1,obj.NumParam,obj.NumVisibleDataStates); %E_v(dE/dth)
            Ex_dEdth_i = zeros(1,obj.NumParam,obj.NumInputDataStates); %E_{v_i}(dE/dth)
            if obj.EstimateHessianFlag
                Ex_dEdth2 = zeros(obj.NumParam,obj.NumParam); %E(dE/dth_i dE/dth_j)
                Ex_dEdth2_v = zeros(obj.NumParam,obj.NumParam,obj.NumVisibleDataStates); %E_v(dE/dth_i dE/dth_j)
                Ex_dEdth2_i = zeros(obj.NumParam,obj.NumParam,obj.NumInputDataStates); %E_{v_i}(dE/dth_i dE/dth_j)
                Cov_dEdth_v = zeros(obj.NumParam,obj.NumParam,obj.NumVisibleDataStates); %Cov_v(dE/dth)
                Cov_dEdth_i = zeros(obj.NumParam,obj.NumParam,obj.NumInputDataStates); %Cov_{v_i}(dE/dth)
            end
            
            %=============================Estimate Expectations=============================
            
            %-------------------------------------------------------------------------------
            %Without conditioning
            %-------------------------------------------------------------------------------
            
            %Update Graph strength
            obj.DWGraph.UpdateStrength(obj.FieldStrength,obj.InteractionStrength,'01')
            %Solve Graph
            obj.DWGraph.SolveGraph();
            %Get solution
            probs = obj.DWGraph.Solution_prob';
            states = obj.DWGraph.Solution_01;
            numStates = length(probs);
            %energy gradient from sampled states
            energy_grad = zeros(numStates,obj.NumParam);
            energy_grad(:,1:obj.NumFieldParam) = states';
            for j=1:obj.NumInteractionParam
                energy_grad(:,obj.NumFieldParam+j) = ...
                    (states(obj.EdgeMap(j,1),:).*states(obj.EdgeMap(j,2),:))';
            end
            %expected values
            Ex_dEdth = probs'*energy_grad;
            if obj.EstimateHessianFlag
                for i=1:obj.NumParam
                    for j=1:obj.NumParam
                        Ex_dEdth2(i,j) =  probs'* (energy_grad(:,i) .*energy_grad(:,j));
                    end
                end
                Cov_dEdth = Ex_dEdth2 - Ex_dEdth'*Ex_dEdth;
            end
            for k = 1:obj.NumVisibleDataStates
                CurrData = obj.VisibleDataStates(:,k);
                temp = sum(abs(states(1:obj.NumVisibleNode,:)...
                    -CurrData*ones(1,numStates)),1) <= 1e-3 ;
                Pr_v(k) = sum(double(temp).*probs');
            end
            Pr_v = 0.9999999*Pr_v + 10^-10*obj.VisibleDataPMF'; %To get finite KLL
            %-------------------------------------------------------------------------------
            %Conditioning on Visible data
            %-------------------------------------------------------------------------------
            
            for k =1:obj.NumVisibleDataStates
                CurrVisibleState = obj.VisibleDataStates(:,k);
                
                %Update Graph strength
                delField = transpose(obj.InteractionStrength(1:obj.NumVisibleNode,...
                    obj.NumVisibleNode+1:obj.NumAllNode))*CurrVisibleState;
                Hnew = obj.FieldStrength(obj.NumVisibleNode+1:obj.NumAllNode) + delField;
                Jnew = obj.InteractionStrength(obj.NumVisibleNode+1:obj.NumAllNode,...
                    obj.NumVisibleNode+1:obj.NumAllNode);
                obj.DWHiddenGraph.UpdateStrength(Hnew,Jnew,'01')
                %Solve Graph
                obj.DWHiddenGraph.SolveGraph();
                %Get solution
                probs_v = obj.DWHiddenGraph.Solution_prob';
                numStates = length(probs_v);
                states_v = [CurrVisibleState*ones(1,numStates); obj.DWHiddenGraph.Solution_01];
                %energy gradient from sampled states
                energy_grad = zeros(numStates,obj.NumParam);
                energy_grad(:,1:obj.NumFieldParam) = states_v';
                for j=1:obj.NumInteractionParam
                    energy_grad(:,obj.NumFieldParam+j) = ...
                            (states_v(obj.EdgeMap(j,1),:).*states_v(obj.EdgeMap(j,2),:))';
                end              
                Ex_dEdth_v(:,:,k) = probs_v'*energy_grad;
                if obj.EstimateHessianFlag
                    for i=1:obj.NumParam
                        for j=1:obj.NumParam
                            Ex_dEdth2_v(i,j,k) =  probs_v'* ...
                                (energy_grad(:,i) .*energy_grad(:,j));
                        end
                    end
                    Cov_dEdth_v(:,:,k) = Ex_dEdth2_v(:,:,k) - Ex_dEdth_v(:,:,k)'*Ex_dEdth_v(:,:,k);
                end
            end
            
            %-------------------------------------------------------------------------------
            %Conditioning on Input data
            %-------------------------------------------------------------------------------
            
            for k =1:obj.NumInputDataStates
                CurrInputState = obj.InputDataStates(:,k);
                
                %Update Graph strength
                delField = transpose(obj.InteractionStrength(1:obj.NumInputNode,obj.NumInputNode+1:obj.NumAllNode))*CurrInputState;
                Hnew = obj.FieldStrength(obj.NumInputNode+1:obj.NumAllNode) + delField;
                Jnew = obj.InteractionStrength(obj.NumInputNode+1:obj.NumAllNode,obj.NumInputNode+1:obj.NumAllNode);
                obj.DWOutputHiddenGraph.UpdateStrength(Hnew,Jnew,'01')
                %Solve Graph
                obj.DWOutputHiddenGraph.SolveGraph();
                %Get solution
                probs_i = obj.DWOutputHiddenGraph.Solution_prob';
                numStates = length(probs_i);
                states_i = [CurrInputState*ones(1,numStates); obj.DWOutputHiddenGraph.Solution_01];
                %energy gradient from sampled states
                energy_grad = zeros(numStates,obj.NumParam);
                energy_grad(:,1:obj.NumFieldParam) = states_i';
                for j=1:obj.NumInteractionParam
                    energy_grad(:,obj.NumFieldParam+j) = ...
                            (states_i(obj.EdgeMap(j,1),:).*states_i(obj.EdgeMap(j,2),:))';
                end                
                Ex_dEdth_i(:,:,k) = probs_i'*energy_grad;
                
                if obj.EstimateHessianFlag
                    for i=1:obj.NumParam
                        for j=1:obj.NumParam
                            Ex_dEdth2_i(i,j,k) =  probs_i'* ...
                                (energy_grad(:,i) .*energy_grad(:,j));
                        end
                    end
                    Cov_dEdth_i(:,:,k) = Ex_dEdth2_i(:,:,k) - Ex_dEdth_i(:,:,k)'*Ex_dEdth_i(:,:,k);
                end
                
                CurrOutputData = obj.VisibleDataStates(obj.NumInputNode+1:obj.NumVisibleNode,k);
                temp = sum(abs(states_i(obj.NumInputNode+1:obj.NumVisibleNode,:)...
                    -CurrOutputData*ones(1,numStates)),1) <= 1e-3 ;
                Pr_v_i(k) = sum(double(temp).*probs_i');
            end
            Pr_v_i = 0.999999*Pr_v_i+ 10^-10*ones(size(Pr_v_i)); %To get finite NegCondLL
            
            %======================Estimate Gradient and Hessian========================
            grad = -obj.CostParameters.Alpha * Ex_dEdth;
            for k=1:obj.NumVisibleDataStates
                const = (obj.CostParameters.Alpha * obj.VisibleDataPMF(k) )+ ...
                    ((1-obj.CostParameters.Alpha)./obj.NumVisibleDataStates);
                grad = grad + const*Ex_dEdth_v(:,:,k);
            end
            for k=1:obj.NumInputDataStates
                const = (1-obj.CostParameters.Alpha)./obj.NumInputDataStates;
                grad = grad - const*Ex_dEdth_i(:,:,k);
            end
            if obj.EstimateHessianFlag
                hess = obj.CostParameters.Alpha * Cov_dEdth;
                for k=1:obj.NumVisibleDataStates
                    const = (obj.CostParameters.Alpha * obj.VisibleDataPMF(k) )+ ...
                        ((1-obj.CostParameters.Alpha)./obj.NumVisibleDataStates);
                    hess = hess - const*Cov_dEdth_v(:,:,k);
                end
                for k=1:obj.NumInputDataStates
                    const = (1-obj.CostParameters.Alpha)./obj.NumInputDataStates;
                    hess = hess + const*Cov_dEdth_i(:,:,k);
                end
            else
                hess=0;
            end
            
            %======================Estimate function value ========================
            
            obj.KLDiv = sum(obj.VisibleDataPMF.*log(obj.VisibleDataPMF./Pr_v'));
            obj.NCondLogLike = -sum(log(Pr_v_i));
            func = obj.CostParameters.Alpha * obj.KLDiv + ...
                ((1-obj.CostParameters.Alpha)./obj.NumInputDataStates)*obj.NCondLogLike;
            
            %======================Update Cost values ========================
            grad= grad';
            obj.CostVal = func;
            obj.GradientVal = grad;
            obj.HessianVal = hess;
            %             min(eig(hess))
        end
        
        function [Hess] = getHessian(obj,~,~)
            %Function to output Hessian during second order learning
            Hess = obj.HessianVal ;
        end
        
        function stop = fminconRunTimeOutput(obj,~,optimValues,state)
            %Function to print results during second order learning
            stop=0;
            if strcmp(state,'iter')
                fprintf('Step: %i, \t F-count: %i, \t KLDiv: %f, \t NCLL: %f, \t Cost: %f, \t StepSize: %f \n'...
                    ,optimValues.iteration, optimValues.funccount,obj.KLDiv,obj.NCondLogLike,...
                    obj.CostVal, optimValues.stepsize);
            end
        end
        
        function CompareDWave(obj)
            %Function to compare KLDivergence and Negative conditional log
            %likelihood of the graph using exact enumeration and DWave
            %---------------Run Exact---------------
            if ~(isobject(obj.States) || isobject(obj.Energy_grad) ||...
                    isobject(obj.VisibleStartIndex) || isobject(obj.InputStartIndex));
            %Create eneumerated States and Energy_grad = (del E)/(del theta)
            SetIndex = 0:obj.NumAllStates-1; %Enumerate all states
            obj.States = ((double(dec2bin(SetIndex'))-48)*2-1)';
            obj.States = (obj.States+1)/2;
            obj.Energy_grad = zeros(length(SetIndex),obj.NumParam);
            obj.Energy_grad(:,1:obj.NumFieldParam) = obj.States';
            for j=1:obj.NumInteractionParam
                obj.Energy_grad(:,obj.NumFieldParam+j) = ...
                    (obj.States(obj.EdgeMap(j,1),:).*obj.States(obj.EdgeMap(j,2),:))';
            end
            %Estimate InputStartIndex
            temp = num2str([obj.InputDataStates',...
                zeros(obj.NumInputDataStates,obj.NumOutputNode + obj.NumHiddenNode)]);
            obj.InputStartIndex = bin2dec(temp)+1;
            %Estimate VisibleStartIndex
            temp = num2str([obj.VisibleDataStates',...
                zeros(obj.NumVisibleDataStates,obj.NumHiddenNode)]);
            obj.VisibleStartIndex = bin2dec(temp)+1;
            end
            obj.ExactBoltzEstimate(obj.CurrPoint,[])
            Exact_KLDiv = obj.KLDiv;
            Exact_NCLL = obj.NCondLogLike;
            %---------------Run DWave ---------------
            if ~(isobject(obj.DWGraph) || isobject(obj.DWHiddenGraph) ...
                    || isobject(obj.DWOutputHiddenGraph))
                obj.CreateDWaveStructure()
            end
            obj.DWaveBoltzEstimate(obj.CurrPoint,[])
            DWave_KLDiv = obj.KLDiv;
            DWave_NCLL = obj.NCondLogLike;
            %---------------Compare ---------------
            fprintf('KLDivergence-\t Exact:%f \t DWave:%f\n',Exact_KLDiv,DWave_KLDiv)
            fprintf('NegCondLogLike-\t Exact:%f \t DWave:%f\n',Exact_NCLL,DWave_NCLL)
        end
        
        function [BetaRange, KLDiv, NCondLogLike] = PlotBMBetaCharacterisctic(obj,BetaRange)
            %Function to plot KLDivergence and Negative conditional log
            %likelihood. 
            %It uses enumeration of states therefore should be used for
            %small graphs
            %Input: 
            % BetaRange: A sequence of beta values (Must be positive and finite)
            
            if ~(isobject(obj.States) || isobject(obj.Energy_grad) ||...
                    isobject(obj.VisibleStartIndex) || isobject(obj.InputStartIndex));
            %Create eneumerated States and Energy_grad = (del E)/(del theta)
            SetIndex = 0:obj.NumAllStates-1; %Enumerate all states
            obj.States = ((double(dec2bin(SetIndex'))-48)*2-1)';
            obj.States = (obj.States+1)/2;
            obj.Energy_grad = zeros(length(SetIndex),obj.NumParam);
            obj.Energy_grad(:,1:obj.NumFieldParam) = obj.States';
            for j=1:obj.NumInteractionParam
                obj.Energy_grad(:,obj.NumFieldParam+j) = ...
                    (obj.States(obj.EdgeMap(j,1),:).*obj.States(obj.EdgeMap(j,2),:))';
            end
            %Estimate InputStartIndex
            temp = num2str([obj.InputDataStates',...
                zeros(obj.NumInputDataStates,obj.NumOutputNode + obj.NumHiddenNode)]);
            obj.InputStartIndex = bin2dec(temp)+1;
            %Estimate VisibleStartIndex
            temp = num2str([obj.VisibleDataStates',...
                zeros(obj.NumVisibleDataStates,obj.NumHiddenNode)]);
            obj.VisibleStartIndex = bin2dec(temp)+1;
            end
            %=============================Estimate Energy=============================
            
            Energy = zeros(obj.NumAllStates,1);
            for i=1:obj.NumAllStates
                Energy(i) = obj.FieldStrength'*obj.States(:,i)  +...
                    (obj.States(:,i)'*obj.InteractionStrength*obj.States(:,i));
            end
            BetaRange = BetaRange(:); 
            BetaRange = sort(BetaRange)';
            
            Z = sum(exp(-Energy*BetaRange),1);
            Pr = exp(-Energy*BetaRange)./(ones(obj.NumAllStates,1)*Z);
            
            Pr_v = zeros(obj.NumVisibleDataStates,length(BetaRange)); %p(.|v)
            Pr_i = zeros(obj.NumInputDataStates,length(BetaRange)); %p(.|v_i)
            %Conditioning on Visible data
            VisibleSize = 2^(obj.NumHiddenNode);
            for k =1:obj.NumVisibleDataStates
                state_subIndex = (obj.VisibleStartIndex(k): ...
                    obj.VisibleStartIndex(k) + VisibleSize-1)' ;
                Pr_v(k,:) = sum(Pr(state_subIndex,:),1);              
            end
            
            %Conditioning on Input data
            InputSize = 2^(obj.NumOutputNode + obj.NumHiddenNode);
            for k =1:obj.NumInputDataStates
                state_subIndex = (obj.InputStartIndex(k): ...
                    obj.InputStartIndex(k) + InputSize-1)' ;
                Pr_i(k,:) = sum(Pr(state_subIndex,:),1);
            end 
            
            temp = obj.VisibleDataPMF'*ones(1,length(BetaRange));
            KLDiv = sum(temp.*log(temp./Pr_v),1);
            NCondLogLike = -sum(log(Pr_v./Pr_i),1);
            figure()
            yyaxis left
            semilogx(BetaRange,KLDiv);
            ylabel('KL Divergence')
            yyaxis right
            semilogx(BetaRange,NCondLogLike);
            ylabel('Negative Conditional Log-Likelihood')
            xlabel('$\beta$','Interpreter','Latex')
        end
        
        function PlotGraph(obj)
            %Funtion to plot the energy parameters as a graph
            G = graph(obj.InteractionStrength,'upper');
            NodeLabel = cell(obj.NumAllNode,1);
            for i=1:obj.NumInputNode
                NodeLabel(i) = {sprintf('Vi%i: %.2f',i,obj.FieldStrength(i))};
            end
            for i=obj.NumInputNode+1 : obj.NumVisibleNode
                NodeLabel(i) = {sprintf('Vo%i: %.2f',i-obj.NumInputNode...
                    ,obj.FieldStrength(i))};
            end
            for i=obj.NumVisibleNode+1 : obj.NumAllNode
                NodeLabel(i) = {sprintf('H%i: %.2f',i-obj.NumVisibleNode...
                    ,obj.FieldStrength(i))};
            end
            h = plot(G,'EdgeLabel',G.Edges.Weight,'LineWidth',2,...
                'NodeLabel',NodeLabel);
            h.MarkerSize = 7;            
            h.Marker = 'o';
            highlight(h,[1:obj.NumInputNode],'NodeColor','r')
            highlight(h,[obj.NumInputNode+1 : obj.NumVisibleNode],'NodeColor','b')
            highlight(h,[obj.NumVisibleNode+1 : obj.NumAllNode],'NodeColor',[17 17 17]/255)

            set(gca,'XTick',[], 'YTick', [])
            daspect([1 1 1])
            set(gca,'color','none')
            set(gca,'Visible','off')
        end
    end
end