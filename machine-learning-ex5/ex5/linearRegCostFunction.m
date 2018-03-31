function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

theta_j = theta; % cause we need unchanged theta too
theta_j(1) = 0; %theta_0 term has little impact
cost_reg_term = (lambda / (2*m)) * sum(theta_j .^ 2);

J = 1/(2*m) * sum((X*theta - y).^ 2 ) + cost_reg_term ;


grad = (X'*(X*theta - y) + lambda*theta_j)/m; %bias term we set to 0


% =========================================================================

grad = grad(:);

end
