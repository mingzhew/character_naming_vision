require 'nn'
require 'paths'
require 'hdf5'

cmd = torch.CmdLine()
cmd:option('-vid' , '')
cmd:option('-crop' , '')
cmd:option('-feat' , '')
params = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
--load net
net = torch.load('vgg_face/VGG_FACE.t7')
net:evaluate()

function load_faces(mv , vid)
	local fn = string.format('%s/%s/%s.h5' , params.crop , mv , vid)
	local file = hdf5.open(fn , 'r')
	local faces = file:read('faces'):all()
	local idx = file:read('idx'):all()
	if faces:dim() ~= 0 then
		faces = faces:permute(1,4,2,3)
	end
	--faces = faces:resize(idx:sum() , 3, 224 ,224)
	print (#faces)
	print (#idx)
	file:close()
	return faces , idx
end

mean = {129.1863,104.7624,93.5940}
function extract_feat(mv , vid)
	local faces, idx = load_faces(mv , vid)
	if faces:dim() == 0 then return nil end
	local trknum = idx:size()[1]
	local feat1 = torch.Tensor(idx:sum() , 4096)
	local feat2 = torch.Tensor(idx:sum() , 4096)
	local cnt = 0
	for i=1,3 do faces[{{},i,{},{}}]:add(-mean[i]) end
	local batch = 100
	while cnt < faces:size()[1] do
		if cnt+batch>faces:size()[1] then batch=faces:size()[1]-cnt end
		face = torch.Tensor(batch,3,224,224):copy(faces[{{cnt+1,cnt+batch}}])
		prob = net(face)
		feat1[{{cnt+1,cnt+batch}}] = net:get(33).output
		feat2[{{cnt+1,cnt+batch}}] = net:get(36).output
		cnt = cnt+batch
		print(i,cnt)
	end
	local dir = string.format('%s/%s' , params.feat , mv)
	local fn = string.format('%s/%s.h5' , dir , vid)
	if paths.dir(dir) == nil then paths.mkdir(dir) end
	local file = hdf5.open(fn , 'w')
	file:write('fc6' , feat1:double())
	file:write('fc7' , feat2:double())
	file:write('idx' , idx)
	file:close()
	print(mv)
end

--load mv
vidpath = params.vid
s = vidpath:split('/')
vid = s[#s]
mv = s[#s-1]
print (mv,vid)
extract_feat(mv , vid)
