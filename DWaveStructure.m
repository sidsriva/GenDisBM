%This class can be used to simulate isinf models in both 0/1 and +1/-1
%format
%Created by Siddhartha Srivastava (sidsriva[at]umich.edu)
%Last update 8/24/2020
classdef DWaveStructure < handle
    properties
        Name %Name of the BM
        Adjacency %Adjacency matrix
        FieldStrength_01=Inf %Field Strength in the 0/1 statespace
        InteractionStrength_01=Inf %Interaction Strength in the 0/1 statespace
        FieldStrength_pm1=Inf %Field Strength in the +1/-1 statespace
        InteractionStrength_pm1=Inf %Interaction Strength in the +1/-1 statespace
        HardwareAdjacency %Adjacency of the solver
        Embedding %Embedding of the Adjacency into Hardware adjacency
        Solver %Choice of DWave solver
        Solution_01=Inf % Solution in 0/1 state space
        Solution_pm1=Inf % Solution in +1/-1 state space
        Solution_prob %Probability of each solution occuring
        Numreads = 100; %Number of expected outputs from 
    end
    methods
        function obj = DWaveStructure(Name, Adjacency,Solver_flag)
            obj.Name = Name;
            obj.Adjacency = Adjacency;
            obj.InitiateDwave(Solver_flag);
        end
        function UpdateStrength(obj,H,J,type) 
            if strcmp(type,'01')
                obj.FieldStrength_01 = H;
                obj.InteractionStrength_01 = J;
                obj.zeroone2pmone();
            elseif strcmp(type,'pm1')
                obj.FieldStrength_pm1 = H;
                obj.InteractionStrength_pm1 = J;
                obj.pmone2zeroone();
            else
                error('Wrong variable type: choose "01" for 0/1 or "pm1" for +1/-1 ');
            end
        end
        function SolveGraph(obj)
            Factor = 1;
            %Use a higher factor if magnitude of parameters is too large.
            %(greater than 2 - based on Dwave manual)
            %Factor = 1*max(max(max(abs(obj.InteractionStrength_pm1))),max(abs(obj.FieldStrength_pm1)));
            J = obj.InteractionStrength_pm1/Factor;
            H = obj.FieldStrength_pm1/Factor;
            %Embed the problem using embedding calculated during initialization
            [h0, j0, jc] = sapiEmbedProblem(H, J, obj.Embedding, obj.HardwareAdjacency);
            lambda = 2;
            %Solve the problem on Dwave
            sample = sapiSolveIsing(obj.Solver, h0, j0 + lambda*jc, 'num_reads', obj.Numreads);
            %Unembed the solution into initial graph
            obj.Solution_pm1 = sapiUnembedAnswer(sample.solutions, obj.Embedding, 'minimize_energy', H*Factor, J*Factor);
            obj.Solution_prob = sample.num_occurrences./ sum(sample.num_occurrences);
            obj.pmone2zeroone(); 
        end
        function [e0] = pmone2zeroone(obj)
            %Converts parameters for Ising from +1/-1 state to parameters
            %for 0/1 states
            %Note J is upper triangular
            %Evaluate +/-1 state energy as:
            % E = H'*S + S'*J*S
            %Evaluate 0/1 state energy as:
            % e = e0+ h'*s + s'*j*s
            H = obj.FieldStrength_pm1;
            J = obj.InteractionStrength_pm1;
            S = obj.Solution_pm1;
            obj.Solution_01 = (S+1)/2;
            obj.InteractionStrength_01 = 4*J;
            obj.FieldStrength_01 = 2*(H - sum(J+J',2));
            e0 = -sum(H) + sum(sum(J));
        end
        function [E0] = zeroone2pmone(obj)
            %Converts parameters for Ising from 0/1 state to parameters
            %for +/-1 states
            %Note J is upper triangular
            %Evaluate 0/1 state energy as:
            % e = h'*s + s'*j*s
            %Evaluate +/-1 state energy as:
            % E = E0+ H'*S + S'*J*S
            h = obj.FieldStrength_01(:);
            j = obj.InteractionStrength_01;
            s = obj.Solution_01;
            obj.Solution_pm1 = 2*s-1;
            obj.InteractionStrength_pm1 = j/4;
            obj.FieldStrength_pm1 = h/2 + (1/4)*sum(j+j',2);
            E0 = +sum(h)/2 + sum(sum(j))/4;
        end
        function InitiateDwave(obj,Solver_flag)
            %this function initialize the dwave solver
            %Input:
            %Nnodes - # of nodes
            %Solver_flag:
            %Solver_flag = 1: Local solver
            %Solver_flag = 2: Remote solver
            
            % Choose Local/Remote connection
            if (Solver_flag==1)
                conn = sapiLocalConnection;
                chip = 'c4-sw_sample';
                obj.Solver = sapiSolver(conn, chip);
            elseif (Solver_flag==2)
                load('login_details_XPS.mat');
                conn = sapiRemoteConnection(url,token);
                chip = 'C16';
                obj.Solver = sapiSolver(conn, chip);
            else
                error('Wrong solver')
            end
            obj.HardwareAdjacency = getHardwareAdjacency(obj.Solver);
            % find and print embeddings
            obj.Embedding = sapiFindEmbedding(obj.Adjacency, obj.HardwareAdjacency, 'verbose', 1);
        end
    end
end