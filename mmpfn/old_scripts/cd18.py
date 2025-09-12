data_name = 'CD18_without_"no-data".xls'
self_data_path = data_path

self.df = pd.read_excel(os.path.join(data_path, data_name), engine='xlrd)
self.cat_cols = ['name', 'Release Date', 'os', 'prcessor', 'Battery_type']
self.numeric_cols = ['Weigth', 'Storage', 'hit', 'hit_count', 'display_size', 'V_resolution', 'H_resolution', 'camera', 'video', 'ram', 'Battery']
self.image_cols = ['Picture']
self.target_col = 'Price'

self.encoder = OrdinalEncoder()
self.x = self.encoder.fit_transform(self.df[self.cat_cols])
self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_cols), self.df[self.numeric_cols]], axis=1).values
self.target_encoder = LabelEncoder() aha?
self.y = self.target_encoder.fit_transform(self.df[self.target_col])


get_images

self.images = []
for i, paths in self.df[self.image_cols].iterrows():
image_set = []
for path in paths:
image_path = os.path.join(self.data_path, 'eimage', path.split('/')[-1])
with Image.open(image_path) as img:
img = img.convert("RGB")
img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32)
image_set.append(img)
self.imgaes.append(image_set)

self.images = np.stack

# get_embeddings petfinder_pawpularity와 동일