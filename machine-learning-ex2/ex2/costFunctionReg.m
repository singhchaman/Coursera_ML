function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


left_term = -y' * log(sigmoid(X*theta));
right_term = (1 - y') * log(1 - sigmoid(X*theta));

theta_j = theta; % cause we need unchanged theta too
theta_j(1) = 0; %theta_0 term has little impact
cost_reg_term = (lambda / (2*m)) * sum(theta_j .^ 2);
J = (1 / m)*(left_term - right_term) + cost_reg_term;

grad_reg_term = (lambda / m) * theta_j;
grad = (1 / m)*(X' * (sigmoid(X*theta) - y)) + grad_reg_term;



% =============================================================

end