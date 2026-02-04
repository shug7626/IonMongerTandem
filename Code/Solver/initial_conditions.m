function sol_init = initial_conditions(psi0,params,vectors,matrices)
% This function creates a vector containing a steady-state solution to the 
% initial problem. The inputs are structures containing the parameters,
% vectors and matrices needed by the solver.

% Parameter input
[chi, nc, pc, Verbose, N, NE, NH, phidisp] = ...
    struct2array(params, {'chi','nc','pc','Verbose','N','NE','NH','phidisp'});
[x, xE, xH] = struct2array(vectors, {'x','xE','xH'});

%% First find initial conditions for the perovskite top-cell at its Vbi
% Define uniform profiles for the ion vacancy density and electric potential
P_init    = ones(size(x));
phi_init  = zeros(size(x))+phidisp;
phiE_init = zeros(size(xE(1:NE)))+phidisp;
phiH_init = zeros(size(xH(1:NH)))+phidisp;

% Compute profiles for the carrier concentrations from a quasi-steady BVP
y_guess = bvpinit(x',@(x) [x+(1-x)/chi; 0*x; chi*x+(1-x); 0*x]);
sol = bvp4c(@(x,y) yode(x,y,params),@(ya,yb) ybcs(ya,yb,params),y_guess);
solx = deval(sol,x'); p_init = solx(1,:)'; n_init = solx(3,:)';

% Define tanh profiles for the carrier concentrations across the TLs
stE = 3/xE(1);
nE_init = nc+(1-nc)*tanh(stE*(xE(1)-xE))/tanh(stE*(xE(1)-xE(end)));
stH = 3/(xH(end)-1);
pH_init = pc+(1-pc)*tanh(stH*(xH(end)-xH))/tanh(stH*(xH(end)-xH(1)));

% Combine the initial conditions into one vector to pass to fsolve
sol_init  = [P_init; phi_init; n_init; p_init; ... % perovskite
             phiE_init; nE_init; ... % electron transport layer
             phiH_init; pH_init]; % hole transport layer

% Define the settings for the call to fsolve
fsoptions = optimoptions('fsolve','MaxIterations',40);
if Verbose, fsoptions.Display = 'iter'; else, fsoptions.Display = 'off'; end

% Use the initial guess to obtain an approximate steady-state solution
if exist('AnJac','file')
    fsoptions.SpecifyObjectiveGradient = true;
    [sol_init,~,exitflag,~] = fsolve(@(u) RHS_AnJac_top(u,psi0, ...
        params,vectors,matrices,'init'),sol_init,fsoptions);
else
    if exist('Jac','file')
        fsoptions.JacobPattern = Jac_top(params,'init');
    end
    [sol_init,~,exitflag,~] = fsolve(@(u) RHS_top(0,u,psi0, ...
        params,vectors,matrices,'init'),sol_init,fsoptions);
end
if exitflag<1
    warning(['Steady-state initial conditions could not be found to ' ...
        'a high degree of accuracy and may be unphysical.']);
end

% Ensure all the algebraic equations are satisfied as exactly as possible
sol_init = apply_Poisson(sol_init,params,vectors,matrices);


%% Append a state for the bottom-cell potential at corresponding current
% Compute and append the potential across the bottom-cell
pbiSi = bottom_cell_potential(sol_init,params,vectors);
sol_init(end+1) = pbiSi;

% Shift the top-cell potential distribution to satisfy the BCs
sol_init([N+2:2*N+2,4*N+5:4*N+NE+4,4*N+2*NE+6:4*N+2*NE+NH+5]) = ...
    sol_init([N+2:2*N+2,4*N+5:4*N+NE+4,4*N+2*NE+6:4*N+2*NE+NH+5])-pbiSi/2;

% Use the initial guess to obtain an approximate steady-state solution
psi0 = @(t) -pbiSi/2;
if exist('AnJac','file')
    fsoptions.SpecifyObjectiveGradient = true;
    [sol_init,~,exitflag,~] = fsolve(@(u) RHS_AnJac(u,psi0, ...
        params,vectors,matrices,'init'),sol_init,fsoptions);
else
    if exist('Jac','file')
        fsoptions.JacobPattern = Jac(params,'init');
    end
    [sol_init,~,exitflag,~] = fsolve(@(u) RHS(0,u,psi0, ...
        params,vectors,matrices,'init'),sol_init,fsoptions);
end
if exitflag<1
    warning(['Steady-state initial conditions could not be found to ' ...
        'a high degree of accuracy and may be unphysical.']);
end

% Ensure all the algebraic equations are satisfied as exactly as possible
sol_init = apply_Poisson(sol_init,params,vectors,matrices);

end


%% Quasi-steady BVP for the carrier concentrations
% y(1) = p(x), y(2) = jp(x) = -Kp*dp/dx, y(3) = n(x), y(4) = jn(x) = Kn*dn/dx
function dpdx = yode(x,y,params)
[G, R, Kp, Kn] = struct2array(params,{'G','R','Kp','Kn'});
dpdx = [-y(2)/Kp; ...
        G(x,0)-R(y(3),y(1),1); ...
        y(4)/Kn; ...
        -(G(x,0)-R(y(3),y(1),1))];
end
function res = ybcs(ya,yb,params)
[Rl, Rr] = struct2array(params,{'Rl','Rr'});
res = [yb(1)-1; ...
       ya(2)+Rl(1,ya(1)); ...
       ya(3)-1; ...
       yb(4)+Rr(yb(3),1)];
end

%% The RHS and the Jacobian for the perovskite top-cell only
function [F, J] = RHS_AnJac_top(u,psi,params,vectors,matrices,flag)
[F, J] = RHS_AnJac([u;0],psi,params,vectors,matrices,flag);
F = F(1:end-1,:);
J = J(1:end-1,1:end-1);
end
function JJJ = Jac_top(params,flag)
JJJ = Jac(params,flag);
JJJ = JJJ(1:end-1,1:end-1);
end
function dudt = RHS_top(t,u,psi,params,vectors,matrices,flag)
dudt = RHS(t,[u;0],psi,params,vectors,matrices,flag);
dudt = dudt(1:end-1,:);
end

%% The potential across the bottom-cell
function pbiSi = bottom_cell_potential(sol_init,params,vectors)
[jay, jsc, j0, nid, VT, Acell, Rp, Rp2, Vbi] = ...
    struct2array(params, {'jay','jsc','j0','nid','VT','Acell','Rp', ...
                          'Rp2','Vbi'});

% Compute the current density through the top-cell and series resistor
dstrbns = unpack(repmat([sol_init;0]',2),params);
params.time = [0,1]; % current is calculated from two time points
[J_top, ~, ~, ~] = calculate_currents(params,vectors,dstrbns);
J_top = J_top(end)*jay; % mA/cm2
J_res = -Vbi/(Acell/1e4*Rp)/10;

% Compute the corresponding potential across the bottom-cell
potential_eqn = @(V) jsc-j0*(exp(V/(nid*VT))-1)-V/(Acell/1e4*Rp2)/10 ...
                     -J_top-J_res;
VSi = fsolve(potential_eqn, Vbi);
pbiSi = VSi/VT;
end
