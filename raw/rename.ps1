$files = Get-ChildItem
$id = 1
$files | foreach { Rename-Item -Path $_.fullname -NewName ( ((($id++).tostring()).padleft(($files.count.tostring()).length) -replace ' ','0' ) + $_.extension) }