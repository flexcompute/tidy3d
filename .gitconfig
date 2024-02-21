[diff "jupyternotebook"]
	command = git-nbdiffdriver diff
[merge "jupyternotebook"]
	driver = git-nbmergedriver merge %O %A %B %L %P
	name = jupyter notebook merge driver
[difftool "nbdime"]
	cmd = git-nbdifftool diff \"$LOCAL\" \"$REMOTE\" \"$BASE\"
[difftool]
	prompt = false
[mergetool "nbdime"]
	cmd = git-nbmergetool merge \"$BASE\" \"$LOCAL\" \"$REMOTE\" \"$MERGED\"
[mergetool]
	prompt = false
