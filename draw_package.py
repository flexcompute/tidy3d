import tidy3d as td
import erdantic as erd

def save_diagram(obj):
	name = obj.__name__
	fname = f'img/diagram_{name}.png'
	model = erd.create(obj)
	model.draw(fname)

def main():
	objects = [td.Simulation, td.Geometry, td.AbstractSource, td.Monitor]
	for obj in objects:
		save_diagram(obj)

if __name__ == '__main__':
	main()