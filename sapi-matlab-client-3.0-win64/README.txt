Getting started with the D-Wave(TM) Quantum Computer Solver API MATLAB Package
==============================================================================

Version 3.0

Add this directory (the one containing the README.txt file you are now
reading) to your MATLAB path using any one of the following methods:

a. Select the File > Set Path... menu item to bring up the Set Path dialog
   box.  Click the "Add Folder..." button, navigate to this directory,
   and click OK.  Click the "Save" button to avoid repeating this process
   every time you start MATLAB. Or,

b. use the "editpath" command in the MATLAB Command Window to bring up the
   Set Path dialog box and proceed as above. Or,

c. use the "addpath <this directory>" command.  You must do this every time
   you start MATLAB.

Once you have updated the MATLAB path, you can do a quick check to make sure
everything is set up properly.  Run this command in the MATLAB Command Window:

>> sapiVersion

It should print the message '3.0'.  If you see a different version
reported or an error occurs, you likely have not set up the MATLAB path
correctly.

You can also check that you can connect to the D-Wave(TM) Quantum Computer
Solver API.  You will need two pieces of information: the SAPI URL and an
authentication token.  The SAPI URL is listed on the "Solver API" page of the
web user interface.  Authentication tokens are also obtained from the web
user interface: click on "API Tokens" in the menu under your user name.  Test
the connection using this command:

>> conn = sapiRemoteConnection(url, token);

If an error occurs, it is likely one of parameters is incorrect.  Otherwise,
you are ready to start using the D-Wave(TM) Quantum Computer Solver API.

You can find some example code in the examples subdirectory.


