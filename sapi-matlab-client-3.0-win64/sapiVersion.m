function version = sapiVersion()
%sapiVersion Get SAPI client version.
%
%  sapiVersion
%  version = sapiVersion()
%
%  Output
%    version: a string containing the current SAPI client version.
%
%  If no output argument is given, the version is printed to the command
%  window.

% Proprietary Information D-Wave Systems Inc.
% Copyright (c) 2015 by D-Wave Systems Inc. All rights reserved.
% Notice this code is licensed to authorized users only under the
% applicable license agreement see eula.txt
% D-Wave Systems Inc., 3033 Beta Ave., Burnaby, BC, V5G 4M9, Canada.

if nargout < 1
  fprintf('%s\n', '3.0')
else
  version  = '3.0';
end
end
