function done = sapiremote_awaitcompletion(phs, mindone, timeout)

% Proprietary Information D-Wave Systems Inc.
% Copyright (c) 2015 by D-Wave Systems Inc. All rights reserved.
% Notice this code is licensed to authorized users only under the
% applicable license agreement see eula.txt
% D-Wave Systems Inc., 3033 Beta Ave., Burnaby, BC, V5G 4M9, Canada.

if ~isnumeric(timeout)
  error('sapiremote:BadArgType', 'timeout must be a number')
end
mindone = min(mindone, numel(phs));
t = tic;
done = sapiremote_mex('awaitcompletion', phs, mindone, min(1, timeout));
while toc(t) < timeout && sum(done) < mindone
  done = sapiremote_mex('awaitcompletion', ...
    phs, mindone, min(1, timeout - toc(t)));
end
end
