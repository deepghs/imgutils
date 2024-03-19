from unittest import skipUnless

import pytest
from hbutils.testing import isolated_directory

from imgutils.sd import read_metadata, save_with_metadata
from test.testings import get_testfile

try:
    import torch
    import safetensors
except (ImportError, ModuleNotFoundError):
    _model_installed = False
else:
    _model_installed = True


@pytest.mark.unittest
class TestSdModel:
    @skipUnless(_model_installed, 'dghs-imgutils[model] required.')
    def test_read_metadata(self):
        metadata = read_metadata(get_testfile('surtr_arknights.safetensors'))
        assert metadata == {
            'modelspec.architecture': 'stable-diffusion-v1/lora',
            'modelspec.date': '2024-03-13T11:54:28',
            'modelspec.encoder_layer': '2',
            'modelspec.implementation': 'diffusers',
            'modelspec.prediction_type': 'epsilon',
            'modelspec.resolution': '512x768',
            'modelspec.sai_model_spec': '1.0.0',
            'modelspec.title': 'surtr_arknights',
            'ss_adaptive_noise_scale': 'None',
            'ss_base_model_version': 'sd_v1',
            'ss_batch_size_per_device': '8',
            'ss_bucket_info': '{"buckets": {"0": {"resolution": [320, 960], "count": 1}, '
                              '"1": {"resolution": [384, 896], "count": 13}, "2": '
                              '{"resolution": [384, 960], "count": 8}, "3": '
                              '{"resolution": [448, 832], "count": 80}, "4": '
                              '{"resolution": [512, 704], "count": 302}, "5": '
                              '{"resolution": [512, 768], "count": 207}, "6": '
                              '{"resolution": [576, 576], "count": 225}, "7": '
                              '{"resolution": [576, 640], "count": 171}, "8": '
                              '{"resolution": [640, 576], "count": 267}, "9": '
                              '{"resolution": [704, 512], "count": 55}, "10": '
                              '{"resolution": [768, 512], "count": 17}, "11": '
                              '{"resolution": [832, 448], "count": 9}, "12": '
                              '{"resolution": [896, 384], "count": 3}}, '
                              '"mean_img_ar_error": 0.03480725602247486}',
            'ss_bucket_no_upscale': 'False',
            'ss_cache_latents': 'True',
            'ss_caption_dropout_every_n_epochs': '0',
            'ss_caption_dropout_rate': '0.05',
            'ss_caption_tag_dropout_rate': '0.0',
            'ss_clip_skip': '2',
            'ss_color_aug': 'False',
            'ss_dataset_dirs': '{"1_1girl": {"n_repeats": 1, "img_count": 1358}}',
            'ss_debiased_estimation': 'False',
            'ss_enable_bucket': 'True',
            'ss_epoch': '24',
            'ss_face_crop_aug_range': 'None',
            'ss_flip_aug': 'False',
            'ss_full_fp16': 'False',
            'ss_gradient_accumulation_steps': '1',
            'ss_gradient_checkpointing': 'True',
            'ss_ip_noise_gamma': 'None',
            'ss_keep_tokens': '1',
            'ss_learning_rate': '0.0002',
            'ss_lowram': 'False',
            'ss_lr_scheduler': 'constant',
            'ss_lr_warmup_steps': '0',
            'ss_max_bucket_reso': '960',
            'ss_max_grad_norm': '1.0',
            'ss_max_token_length': '150',
            'ss_max_train_steps': '4224',
            'ss_min_bucket_reso': '320',
            'ss_min_snr_gamma': '0',
            'ss_mixed_precision': 'fp16',
            'ss_multires_noise_discount': '0.3',
            'ss_multires_noise_iterations': '6',
            'ss_network_alpha': '2',
            'ss_network_args': '{"preset": "attn-mlp", "algo": "lora", "dropout": 0}',
            'ss_network_dim': '4',
            'ss_network_dropout': '0',
            'ss_network_module': 'lycoris.kohya',
            'ss_new_sd_model_hash': 'a7529df02340e5b4c3870c894c1ae84f22ea7b37fd0633e5bacfad9618228032',
            'ss_noise_offset': 'None',
            'ss_num_batches_per_epoch': '176',
            'ss_num_epochs': '24',
            'ss_num_reg_images': '0',
            'ss_num_train_images': '1358',
            'ss_optimizer': 'bitsandbytes.optim.adamw.AdamW8bit(weight_decay=0.1,betas=(0.9, '
                            '0.99))',
            'ss_output_name': 'surtr_arknights',
            'ss_prior_loss_weight': '1.0',
            'ss_random_crop': 'False',
            'ss_reg_dataset_dirs': '{}',
            'ss_resolution': '(512, 768)',
            'ss_scale_weight_norms': 'None',
            'ss_sd_model_hash': 'e6e8e1fc',
            'ss_sd_model_name': 'nai.ckpt',
            'ss_sd_scripts_commit_hash': '2d7389185c021bc527b414563c245c5489d6328a',
            'ss_seed': '23',
            'ss_session_id': '1696480111',
            'ss_shuffle_caption': 'True',
            'ss_steps': '4224',
            'ss_tag_frequency': '{"1_1girl": {"surtr (arknights)": 1358, "solo": 1355, '
                                '"1girl": 1357, "purple eyes": 1339, "red hair": 1358, '
                                '"black dress": 683, "bare shoulders": 1076, "looking at '
                                'viewer": 1268, "horns": 671, "very long hair": 562, '
                                '"white background": 375, "off shoulder": 421, "black '
                                'thighhighs": 185, "popsicle": 38, "holding food": 42, '
                                '"standing": 86, "short dress": 23, "bangs": 1276, '
                                '"character name": 37, "simple background": 399, "medium '
                                'breasts": 543, "hair between eyes": 943, "thighs": 92, '
                                '"zettai ryouiki": 20, "long sleeves": 272, "feet out of '
                                'frame": 46, "cleavage": 703, "smile": 333, "chest '
                                'strap": 501, "black jacket": 228, "hair intakes": 300, '
                                '"armband": 191, "upper body": 786, "black belt": 138, '
                                '"grey background": 83, "detached collar": 382, "long '
                                'hair": 795, "portrait": 235, "gradient background": 57, '
                                '"demon horns": 681, "open jacket": 102, "parted lips": '
                                '257, "hand up": 112, "v-shaped eyebrows": 51, "black '
                                'background": 63, "slit pupils": 235, "red background": '
                                '15, "demon girl": 356, "gradient": 53, "jacket": 141, '
                                '"belt": 98, "id card": 78, "closed mouth": 350, "open '
                                'clothes": 38, "ice cream": 33, "dress": 14, "sleeveless '
                                'dress": 164, "sitting": 161, "sleeveless": 66, "cowboy '
                                'shot": 116, "infection monitor \\\\(arknights\\\\)": 40, '
                                '"holding sword": 84, "pouch": 19, "ahoge": 3, "arm up": '
                                '129, "red skirt": 105, "black shirt": 217, "official '
                                'alternate costume": 343, "short sleeves": 171, "black '
                                'gloves": 124, "miniskirt": 56, "cross necklace": 138, '
                                '"black choker": 269, "crown": 19, "thigh strap": 127, '
                                '"chain": 68, "hand on hip": 14, "sword": 145, "lying": '
                                '8, "leaning forward": 9, "black footwear": 94, "watson '
                                'cross": 1, "full body": 52, "shorts": 5, "large '
                                'breasts": 281, "boots": 2, "midriff": 115, "torn '
                                'clothes": 8, "navel": 170, "handcuffs": 2, "garter '
                                'straps": 7, "looking back": 58, "from behind": 36, "back '
                                'focus": 1, "breasts": 109, "detached sleeves": 126, '
                                '"back": 12, "braid": 4, "knee up": 22, "high heels": 64, '
                                '"looking to the side": 27, "backless dress": 14, "white '
                                'footwear": 1, "alternate costume": 96, "signature": 11, '
                                '"dated": 8, "on table": 1, "shadow": 2, "choker": 41, '
                                '"artist name": 18, "from side": 79, "backless outfit": '
                                '3, "pantyhose": 12, "armpits": 120, "black panties": 18, '
                                '"arms up": 63, "torn dress": 4, "highleg": 1, "blush": '
                                '128, "fire": 43, "holding weapon": 51, "ice cream cone": '
                                '13, ":o": 3, "open mouth": 45, "toenail polish": 4, '
                                '"toes": 5, "black nails": 32, "black socks": 25, '
                                '"barefoot": 15, "foot focus": 1, "shoes removed": 3, '
                                '"single glove": 46, "planted": 1, "soles": 2, "hand in '
                                'own hair": 19, "collar": 67, "kneehighs": 21, "one side '
                                'up": 16, "jewelry": 9, "foreshortening": 1, "ponytail": '
                                '142, "collarbone": 123, "black collar": 14, "animal '
                                'skull": 2, "holding skull": 2, "white dress": 17, "pink '
                                'hair": 1, "gauntlets": 1, "nail polish": 64, "necklace": '
                                '79, "skull": 1, "holding": 82, "hair over one eye": 8, '
                                '"pink nails": 2, "spiked collar": 11, "glasses": 2, '
                                '"skateboard": 14, "eyewear removed": 2, "holding '
                                'eyewear": 2, "midriff peek": 10, "shoes": 20, "pencil '
                                'skirt": 19, "t-shirt": 1, "sunglasses": 21, "red nails": '
                                '34, "fingerless gloves": 33, "thighhighs": 18, "wings": '
                                '2, "blue sky": 67, "day": 76, "outdoors": 92, "star '
                                '\\\\(symbol\\\\)": 99, "hair rings": 98, "water": 53, '
                                '"star hair ornament": 86, "cloud": 28, "bikini": 73, '
                                '"swimsuit cover-up": 45, "partially submerged": 3, "blue '
                                'background": 6, "side-tie bikini bottom": 38, ":q": 7, '
                                '"stomach": 106, "criss-cross halter": 105, "black '
                                'bikini": 78, "gloves": 26, "spikes": 7, "weapon": 36, '
                                '":t": 3, "pout": 3, "ass": 7, "earrings": 34, '
                                '"underboob": 12, "holding hair": 6, "orange background": '
                                '7, "own hands together": 1, "torn thighhighs": 4, "head '
                                'tilt": 25, "molten rock": 57, "cross": 41, "crop top": '
                                '77, "holding can": 4, "crossed legs": 24, "graffiti": '
                                '10, "black ribbon": 9, "spray can": 5, "soda can": 2, '
                                '"hammer": 1, "zipper": 11, "one eye closed": 18, '
                                '"holding cup": 24, "indoors": 37, "strapless dress": 11, '
                                '"checkered floor": 4, "side slit": 1, "bare legs": 14, '
                                '"bare arms": 51, "cocktail dress": 3, "legs": 8, "purple '
                                'dress": 3, "ear piercing": 7, "bridal gauntlets": 8, '
                                '"panties": 1, "wading": 7, "seagull": 3, "ocean": 11, '
                                '"bird": 3, "swimsuit": 3, "cross earrings": 3, "crossed '
                                'arms": 7, "embers": 4, "nude": 5, "bandaid on hand": 2, '
                                '"red jacket": 4, "stairs": 1, "bandaid on arm": 2, '
                                '"ring": 28, "socks": 2, "bandaid on knee": 1, '
                                '"asymmetrical legwear": 1, "single thighhigh": 2, "white '
                                'shirt": 15, "loafers": 1, "serafuku": 4, "bandaged leg": '
                                '1, "bandaid on face": 5, "injury": 1, "plaid bow": 2, '
                                '"pleated skirt": 3, "black skirt": 5, "blue bowtie": 1, '
                                '"hair bow": 3, "alternate hairstyle": 9, "blood": 5, '
                                '"black sailor collar": 1, "aqua bow": 1, "bandages": 4, '
                                '"blue bow": 2, "sailor collar": 8, "bowtie": 1, "bandaid '
                                'on cheek": 2, "from below": 6, "mouth hold": 65, '
                                '"spoon": 1, "table": 7, "highleg panties": 2, "hand on '
                                'own cheek": 4, "blurry": 3, "buckle": 3, "mini crown": '
                                '7, "hand on own face": 6, "off-shoulder dress": 3, '
                                '"shirt lift": 2, "grey panties": 1, "skirt pull": 1, '
                                '"paint splatter": 3, "lifted by self": 3, "cup": 6, '
                                '"hammock": 13, "bracelet": 31, "holding spoon": 7, "bead '
                                'bracelet": 1, "umbrella": 1, "on back": 28, "frilled '
                                'bikini": 1, "black necktie": 3, "necktie between '
                                'breasts": 1, "food": 12, "between breasts": 1, "plant": '
                                '5, "on stomach": 3, "red shorts": 2, "short shorts": 3, '
                                '"bag": 7, "skirt": 4, "disposable cup": 2, "hand in '
                                'pocket": 3, "coat": 1, "casual": 4, "bubble tea": 1, '
                                '"floating hair": 9, "curtains": 3, "white panties": 1, '
                                '"window": 7, "off-shoulder sweater": 2, "grey sweater": '
                                '2, "sleeves past wrists": 2, "hair over shoulder": 2, '
                                '"black sweater": 2, "zipper pull tab": 5, "cat": 1, '
                                '"food in mouth": 12, "black one-piece swimsuit": 2, '
                                '"pool": 1, "profile": 24, "blue hair": 1, "2girls": 1, '
                                '"poolside": 1, "solo focus": 4, "phone screen": 1, '
                                '"cherry": 30, "hair ornament": 14, "petals": 19, "burnt '
                                'clothes": 23, "weibo username": 14, "blurry background": '
                                '4, ":3": 7, "sidelocks": 7, "cropped torso": 7, '
                                '"lollipop": 4, "sideboob": 23, "chain-link fence": 3, '
                                '"feet": 11, "no shoes": 5, "sweat": 6, "kneeling": 3, '
                                '"fang": 3, "front-tie top": 3, "food on body": 1, '
                                '"front-tie bikini top": 8, "hands up": 11, "stool": 3, '
                                '"drinking glass": 7, "red footwear": 1, "white gloves": '
                                '4, "sign": 1, "sky": 18, "orange sky": 2, "sunset": 5, '
                                '"evening": 1, "butterfly": 2, "butterfly on hand": 1, '
                                '"holding shoes": 2, "sandals": 6, "flower": 11, "sun '
                                'hat": 4, "straw hat": 3, "yokozuwari": 1, "headwear '
                                'removed": 1, "ribbon": 2, "see-through": 10, "holding '
                                'flower": 2, "halterneck": 12, "red dress": 4, "handbag": '
                                '2, "cropped jacket": 2, "single hair bun": 3, "red '
                                'shirt": 2, "covered navel": 7, "holding bottle": 2, '
                                '"soda bottle": 1, "coca-cola": 2, "vending machine": 2, '
                                '"tying hair": 6, "hair tie in mouth": 6, "playing '
                                'instrument": 3, "violin": 3, "holding instrument": 5, '
                                '"pink flower": 3, "knees up": 14, "arm behind head": 10, '
                                '"tongue out": 11, "cigarette": 2, "groin": 8, "blood on '
                                'face": 12, "arms behind head": 6, "red necktie": 2, '
                                '"collared shirt": 5, "grey shirt": 1, "red rose": 5, '
                                '"vest": 2, "rose": 5, "red flower": 3, "armlet": 3, '
                                '"wet": 10, "close-up": 6, "arm strap": 10, "stage": 2, '
                                '"electric guitar": 2, "concert": 1, "stage lights": 2, '
                                '"music": 1, "glowstick": 1, "guitar": 1, "from above": '
                                '7, "looking up": 6, "red panties": 1, "eating": 1, '
                                '"bra": 3, "underwear only": 4, "champagne flute": 4, '
                                '"bare back": 9, "small breasts": 1, "black serafuku": 5, '
                                '"red neckerchief": 5, "crop top overhang": 3, '
                                '"toenails": 1, "chromatic aberration": 6, "yellow '
                                'background": 4, "hand on own head": 1, "arm support": 3, '
                                '"dutch angle": 6, "lipstick tube": 1, "lipstick": 2, '
                                '"red lips": 5, "purple nails": 2, "lips": 8, "enmaided": '
                                '3, "white apron": 2, "frills": 2, "black pants": 2, '
                                '"sneakers": 1, "hip vent": 1, "cleavage cutout": 1, '
                                '"white bow": 1, "nose blush": 3, "shallow water": 1, '
                                '"petals on liquid": 1, "white bikini": 9, "sideways '
                                'glance": 9, "copyright name": 2, "hand on another\'s '
                                'face": 1, "red choker": 4, "babydoll": 1, "untied '
                                'panties": 1, "side-tie panties": 1, "white pantyhose": '
                                '1, "pointy ears": 2, "hand on own stomach": 1, "areola '
                                'slip": 2, "underbust": 3, "pantyshot": 1, "white pants": '
                                '2, "blue eyes": 12, "shirt": 2, "grin": 8, "red eyes": '
                                '12, "playboy bunny": 8, "black bowtie": 3, "rabbit '
                                'ears": 6, "black leotard": 6, "strapless leotard": 4, '
                                '"fake animal ears": 6, "thigh gap": 5, "black '
                                'pantyhose": 10, "brown pantyhose": 2, "fishnet '
                                'pantyhose": 2, "leotard": 2, "chain necklace": 4, '
                                '"beach": 12, "on couch": 4, "couch": 8, "halter dress": '
                                '4, "pelvic curtain": 2, "breasts apart": 3, "plunging '
                                'neckline": 1, "pocky": 3, ";\\\\)": 2, "tarot": 1, '
                                '"yellow flower": 2, "bubble": 2, "underwater": 1, "water '
                                'drop": 1, "hands in hair": 3, "eyewear on head": 17, '
                                '"red-tinted eyewear": 3, "palm tree": 8, '
                                '"skindentation": 4, "cameltoe": 2, "sunlight": 6, '
                                '"string bikini": 1, "orange nails": 1, "hand on own '
                                'chest": 8, "strapless": 5, "white choker": 6, "elbow '
                                'gloves": 13, "piercing": 1, "head out of frame": 3, '
                                '"covered nipples": 4, "looking away": 5, "sundae": 1, '
                                '"wall clock": 1, "parfait": 1, "fruit": 1, "o-ring": 2, '
                                '"black bow": 1, "no panties": 1, "bow": 2, "wine glass": '
                                '9, "chest sarashi": 2, "bandeau": 1, "fishnet gloves": '
                                '1, "fishnets": 1, "argyle background": 5, "light '
                                'particles": 1, "holding fruit": 8, "apple": 6, "ass '
                                'visible through thighs": 3, "shoe soles": 1, "paw '
                                'print": 1, "bed sheet": 2, "licking": 3, "innertube": 7, '
                                '"off-shoulder shirt": 3, "ball": 1, "bikini under '
                                'clothes": 3, "lily pad": 1, "wristband": 1, "two-tone '
                                'background": 2, "giant": 1, "indian style": 1, "black '
                                'bra": 4, "electric fan": 1, "head rest": 2, "purple '
                                'background": 2, "heart": 1, "sunbeam": 5, "light rays": '
                                '1, "outstretched arms": 1, "crow": 1, "arms under '
                                'breasts": 1, "breast hold": 2, "pink eyes": 2, "potted '
                                'plant": 4, "handheld game console": 2, "nintendo '
                                'switch": 2, "boots removed": 1, "blinds": 1, "single '
                                'shoe": 1, "dress lift": 1, "sweatdrop": 2, "one eye '
                                'covered": 2, "rock": 3, "hat": 1, "closed eyes": 6, "red '
                                'theme": 1, "wrist cuffs": 3, "bar \\\\(place\\\\)": 3, '
                                '"bottle": 2, "alcohol": 2, "red bowtie": 3, "pink bow": '
                                '1, "spaghetti strap": 1, "black shorts": 3, "hair in '
                                'mouth": 1, "zoom layer": 2, "crazy straw": 1, "beach '
                                'umbrella": 4, "lens flare": 5, "beach chair": 2, "o-ring '
                                'bikini": 5, "tropical drink": 1, "hand on eyewear": 2, '
                                '"chair": 1, "adjusting eyewear": 5, "glint": 1, "sand": '
                                '1, "belt buckle": 2, "ground vehicle": 1, "car": 1, '
                                '"building": 2, "city": 1, "mountain": 1, "fangs": 3, '
                                '"teeth": 4, ":d": 3, "chewing gum": 13, "bubble '
                                'blowing": 12, "black rose": 3, "twintails": 2, "holding '
                                'phone": 5, "red ribbon": 6, "hairclip": 3, "neck '
                                'ribbon": 3, "layered dress": 1, "hair flower": 3, "chess '
                                'piece": 2, "red bow": 2, "king \\\\(chess\\\\)": 1, '
                                '"white rose": 1, "phone": 1, "rubber duck": 4, "saliva": '
                                '3, "snot": 1, "belt pouch": 4, "grey dress": 2, "wine": '
                                '7, "string of fate": 2, "dress pull": 2, "disembodied '
                                'limb": 1, "clenched teeth": 2, "dress tug": 1, '
                                '"smartphone": 3, "neckerchief": 1, "cherry blossoms": 3, '
                                '"maid headdress": 3, "wine bottle": 1, "maid": 2, "legs '
                                'up": 1, "brown thighhighs": 2, "partially fingerless '
                                'gloves": 1, "tied shirt": 2, "see-through shirt": 1, '
                                '"standing on one leg": 1, "horns through headwear": 2, '
                                '"red bikini": 2, "hand on headwear": 1, "heterochromia": '
                                '1, "beachball": 1, "sun": 1, "brown headwear": 1, '
                                '"fence": 1, "frown": 1, "bandaged arm": 1, ":<": 1, '
                                '"sleeveless shirt": 1, "short hair": 1, "medium hair": '
                                '1, "purple thighhighs": 1, "tongue": 2, "hand on own '
                                'thigh": 1, "white thighhighs": 1, "shading eyes": 2, '
                                '"foot out of frame": 1, "rainbow": 1, "glowing eyes": 2, '
                                '"garter belt": 2, "lingerie": 2, "star \\\\(sky\\\\)": '
                                '3, "starry sky": 3, "night": 2, "night sky": 1, "moon": '
                                '3, "black vest": 2, "weibo logo": 1, "border": 1, '
                                '"tinted eyewear": 3, ":p": 9, "one knee": 1, "white '
                                'shorts": 1, "mug": 3, "pajamas": 1, "strap slip": 2, '
                                '"tiger": 1, "tattoo": 4, "spade \\\\(shape\\\\)": 3, '
                                '"yellow eyes": 2, "soaking feet": 1, "looking over '
                                'eyewear": 3, "torn shirt": 1, "upside-down": 1, "holding '
                                'baseball bat": 1, "baseball bat": 1, "stretching": 3, '
                                '"wavy mouth": 1, "tears": 1, "highleg leotard": 2, '
                                '"purple hair": 2, "puffy sleeves": 1, "no pants": 1, '
                                '"halo": 1, "rhodes island logo": 1, "bokeh": 2, "center '
                                'opening": 1}}',
            'ss_text_encoder_lr': '0.0006',
            'ss_total_batch_size': '8',
            'ss_training_comment': 'nebulae',
            'ss_training_finished_at': '1710330868.748919',
            'ss_training_started_at': '1710328450.662775',
            'ss_unet_lr': '0.0006',
            'ss_v2': 'False',
            'ss_zero_terminal_snr': 'False',
            'sshs_legacy_hash': '147fd55d',
            'sshs_model_hash': '06adff80a2d8188d3d96c7629d360bd074034ed60da6dfb0752bd31845a21ac8'
        }

    @skipUnless(_model_installed, 'dghs-imgutils[model] required.')
    def test_save_metadata_clear(self):
        with isolated_directory({'surtr_arknights.safetensors': get_testfile('surtr_arknights.safetensors')}):
            save_with_metadata(
                src_model_file='surtr_arknights.safetensors',
                dst_model_file='new.safetensors',
                metadata={'f': '123 surtr'},
                clear=True
            )
            assert read_metadata('new.safetensors') == {'f': '123 surtr'}

    @skipUnless(_model_installed, 'dghs-imgutils[model] required.')
    def test_save_metadata_non_clear(self):
        with isolated_directory({'surtr_arknights.safetensors': get_testfile('surtr_arknights.safetensors')}):
            save_with_metadata(
                src_model_file='surtr_arknights.safetensors',
                dst_model_file='new.safetensors',
                metadata={
                    'f': '123 surtr',
                    'sshs_model_hash': 'bullshit'
                },
            )
            assert read_metadata('new.safetensors') == {
                'f': '123 surtr',
                'modelspec.architecture': 'stable-diffusion-v1/lora',
                'modelspec.date': '2024-03-13T11:54:28',
                'modelspec.encoder_layer': '2',
                'modelspec.implementation': 'diffusers',
                'modelspec.prediction_type': 'epsilon',
                'modelspec.resolution': '512x768',
                'modelspec.sai_model_spec': '1.0.0',
                'modelspec.title': 'surtr_arknights',
                'ss_adaptive_noise_scale': 'None',
                'ss_base_model_version': 'sd_v1',
                'ss_batch_size_per_device': '8',
                'ss_bucket_info': '{"buckets": {"0": {"resolution": [320, 960], "count": 1}, '
                                  '"1": {"resolution": [384, 896], "count": 13}, "2": '
                                  '{"resolution": [384, 960], "count": 8}, "3": '
                                  '{"resolution": [448, 832], "count": 80}, "4": '
                                  '{"resolution": [512, 704], "count": 302}, "5": '
                                  '{"resolution": [512, 768], "count": 207}, "6": '
                                  '{"resolution": [576, 576], "count": 225}, "7": '
                                  '{"resolution": [576, 640], "count": 171}, "8": '
                                  '{"resolution": [640, 576], "count": 267}, "9": '
                                  '{"resolution": [704, 512], "count": 55}, "10": '
                                  '{"resolution": [768, 512], "count": 17}, "11": '
                                  '{"resolution": [832, 448], "count": 9}, "12": '
                                  '{"resolution": [896, 384], "count": 3}}, '
                                  '"mean_img_ar_error": 0.03480725602247486}',
                'ss_bucket_no_upscale': 'False',
                'ss_cache_latents': 'True',
                'ss_caption_dropout_every_n_epochs': '0',
                'ss_caption_dropout_rate': '0.05',
                'ss_caption_tag_dropout_rate': '0.0',
                'ss_clip_skip': '2',
                'ss_color_aug': 'False',
                'ss_dataset_dirs': '{"1_1girl": {"n_repeats": 1, "img_count": 1358}}',
                'ss_debiased_estimation': 'False',
                'ss_enable_bucket': 'True',
                'ss_epoch': '24',
                'ss_face_crop_aug_range': 'None',
                'ss_flip_aug': 'False',
                'ss_full_fp16': 'False',
                'ss_gradient_accumulation_steps': '1',
                'ss_gradient_checkpointing': 'True',
                'ss_ip_noise_gamma': 'None',
                'ss_keep_tokens': '1',
                'ss_learning_rate': '0.0002',
                'ss_lowram': 'False',
                'ss_lr_scheduler': 'constant',
                'ss_lr_warmup_steps': '0',
                'ss_max_bucket_reso': '960',
                'ss_max_grad_norm': '1.0',
                'ss_max_token_length': '150',
                'ss_max_train_steps': '4224',
                'ss_min_bucket_reso': '320',
                'ss_min_snr_gamma': '0',
                'ss_mixed_precision': 'fp16',
                'ss_multires_noise_discount': '0.3',
                'ss_multires_noise_iterations': '6',
                'ss_network_alpha': '2',
                'ss_network_args': '{"preset": "attn-mlp", "algo": "lora", "dropout": 0}',
                'ss_network_dim': '4',
                'ss_network_dropout': '0',
                'ss_network_module': 'lycoris.kohya',
                'ss_new_sd_model_hash': 'a7529df02340e5b4c3870c894c1ae84f22ea7b37fd0633e5bacfad9618228032',
                'ss_noise_offset': 'None',
                'ss_num_batches_per_epoch': '176',
                'ss_num_epochs': '24',
                'ss_num_reg_images': '0',
                'ss_num_train_images': '1358',
                'ss_optimizer': 'bitsandbytes.optim.adamw.AdamW8bit(weight_decay=0.1,betas=(0.9, '
                                '0.99))',
                'ss_output_name': 'surtr_arknights',
                'ss_prior_loss_weight': '1.0',
                'ss_random_crop': 'False',
                'ss_reg_dataset_dirs': '{}',
                'ss_resolution': '(512, 768)',
                'ss_scale_weight_norms': 'None',
                'ss_sd_model_hash': 'e6e8e1fc',
                'ss_sd_model_name': 'nai.ckpt',
                'ss_sd_scripts_commit_hash': '2d7389185c021bc527b414563c245c5489d6328a',
                'ss_seed': '23',
                'ss_session_id': '1696480111',
                'ss_shuffle_caption': 'True',
                'ss_steps': '4224',
                'ss_tag_frequency': '{"1_1girl": {"surtr (arknights)": 1358, "solo": 1355, '
                                    '"1girl": 1357, "purple eyes": 1339, "red hair": 1358, '
                                    '"black dress": 683, "bare shoulders": 1076, "looking at '
                                    'viewer": 1268, "horns": 671, "very long hair": 562, '
                                    '"white background": 375, "off shoulder": 421, "black '
                                    'thighhighs": 185, "popsicle": 38, "holding food": 42, '
                                    '"standing": 86, "short dress": 23, "bangs": 1276, '
                                    '"character name": 37, "simple background": 399, "medium '
                                    'breasts": 543, "hair between eyes": 943, "thighs": 92, '
                                    '"zettai ryouiki": 20, "long sleeves": 272, "feet out of '
                                    'frame": 46, "cleavage": 703, "smile": 333, "chest '
                                    'strap": 501, "black jacket": 228, "hair intakes": 300, '
                                    '"armband": 191, "upper body": 786, "black belt": 138, '
                                    '"grey background": 83, "detached collar": 382, "long '
                                    'hair": 795, "portrait": 235, "gradient background": 57, '
                                    '"demon horns": 681, "open jacket": 102, "parted lips": '
                                    '257, "hand up": 112, "v-shaped eyebrows": 51, "black '
                                    'background": 63, "slit pupils": 235, "red background": '
                                    '15, "demon girl": 356, "gradient": 53, "jacket": 141, '
                                    '"belt": 98, "id card": 78, "closed mouth": 350, "open '
                                    'clothes": 38, "ice cream": 33, "dress": 14, "sleeveless '
                                    'dress": 164, "sitting": 161, "sleeveless": 66, "cowboy '
                                    'shot": 116, "infection monitor \\\\(arknights\\\\)": 40, '
                                    '"holding sword": 84, "pouch": 19, "ahoge": 3, "arm up": '
                                    '129, "red skirt": 105, "black shirt": 217, "official '
                                    'alternate costume": 343, "short sleeves": 171, "black '
                                    'gloves": 124, "miniskirt": 56, "cross necklace": 138, '
                                    '"black choker": 269, "crown": 19, "thigh strap": 127, '
                                    '"chain": 68, "hand on hip": 14, "sword": 145, "lying": '
                                    '8, "leaning forward": 9, "black footwear": 94, "watson '
                                    'cross": 1, "full body": 52, "shorts": 5, "large '
                                    'breasts": 281, "boots": 2, "midriff": 115, "torn '
                                    'clothes": 8, "navel": 170, "handcuffs": 2, "garter '
                                    'straps": 7, "looking back": 58, "from behind": 36, "back '
                                    'focus": 1, "breasts": 109, "detached sleeves": 126, '
                                    '"back": 12, "braid": 4, "knee up": 22, "high heels": 64, '
                                    '"looking to the side": 27, "backless dress": 14, "white '
                                    'footwear": 1, "alternate costume": 96, "signature": 11, '
                                    '"dated": 8, "on table": 1, "shadow": 2, "choker": 41, '
                                    '"artist name": 18, "from side": 79, "backless outfit": '
                                    '3, "pantyhose": 12, "armpits": 120, "black panties": 18, '
                                    '"arms up": 63, "torn dress": 4, "highleg": 1, "blush": '
                                    '128, "fire": 43, "holding weapon": 51, "ice cream cone": '
                                    '13, ":o": 3, "open mouth": 45, "toenail polish": 4, '
                                    '"toes": 5, "black nails": 32, "black socks": 25, '
                                    '"barefoot": 15, "foot focus": 1, "shoes removed": 3, '
                                    '"single glove": 46, "planted": 1, "soles": 2, "hand in '
                                    'own hair": 19, "collar": 67, "kneehighs": 21, "one side '
                                    'up": 16, "jewelry": 9, "foreshortening": 1, "ponytail": '
                                    '142, "collarbone": 123, "black collar": 14, "animal '
                                    'skull": 2, "holding skull": 2, "white dress": 17, "pink '
                                    'hair": 1, "gauntlets": 1, "nail polish": 64, "necklace": '
                                    '79, "skull": 1, "holding": 82, "hair over one eye": 8, '
                                    '"pink nails": 2, "spiked collar": 11, "glasses": 2, '
                                    '"skateboard": 14, "eyewear removed": 2, "holding '
                                    'eyewear": 2, "midriff peek": 10, "shoes": 20, "pencil '
                                    'skirt": 19, "t-shirt": 1, "sunglasses": 21, "red nails": '
                                    '34, "fingerless gloves": 33, "thighhighs": 18, "wings": '
                                    '2, "blue sky": 67, "day": 76, "outdoors": 92, "star '
                                    '\\\\(symbol\\\\)": 99, "hair rings": 98, "water": 53, '
                                    '"star hair ornament": 86, "cloud": 28, "bikini": 73, '
                                    '"swimsuit cover-up": 45, "partially submerged": 3, "blue '
                                    'background": 6, "side-tie bikini bottom": 38, ":q": 7, '
                                    '"stomach": 106, "criss-cross halter": 105, "black '
                                    'bikini": 78, "gloves": 26, "spikes": 7, "weapon": 36, '
                                    '":t": 3, "pout": 3, "ass": 7, "earrings": 34, '
                                    '"underboob": 12, "holding hair": 6, "orange background": '
                                    '7, "own hands together": 1, "torn thighhighs": 4, "head '
                                    'tilt": 25, "molten rock": 57, "cross": 41, "crop top": '
                                    '77, "holding can": 4, "crossed legs": 24, "graffiti": '
                                    '10, "black ribbon": 9, "spray can": 5, "soda can": 2, '
                                    '"hammer": 1, "zipper": 11, "one eye closed": 18, '
                                    '"holding cup": 24, "indoors": 37, "strapless dress": 11, '
                                    '"checkered floor": 4, "side slit": 1, "bare legs": 14, '
                                    '"bare arms": 51, "cocktail dress": 3, "legs": 8, "purple '
                                    'dress": 3, "ear piercing": 7, "bridal gauntlets": 8, '
                                    '"panties": 1, "wading": 7, "seagull": 3, "ocean": 11, '
                                    '"bird": 3, "swimsuit": 3, "cross earrings": 3, "crossed '
                                    'arms": 7, "embers": 4, "nude": 5, "bandaid on hand": 2, '
                                    '"red jacket": 4, "stairs": 1, "bandaid on arm": 2, '
                                    '"ring": 28, "socks": 2, "bandaid on knee": 1, '
                                    '"asymmetrical legwear": 1, "single thighhigh": 2, "white '
                                    'shirt": 15, "loafers": 1, "serafuku": 4, "bandaged leg": '
                                    '1, "bandaid on face": 5, "injury": 1, "plaid bow": 2, '
                                    '"pleated skirt": 3, "black skirt": 5, "blue bowtie": 1, '
                                    '"hair bow": 3, "alternate hairstyle": 9, "blood": 5, '
                                    '"black sailor collar": 1, "aqua bow": 1, "bandages": 4, '
                                    '"blue bow": 2, "sailor collar": 8, "bowtie": 1, "bandaid '
                                    'on cheek": 2, "from below": 6, "mouth hold": 65, '
                                    '"spoon": 1, "table": 7, "highleg panties": 2, "hand on '
                                    'own cheek": 4, "blurry": 3, "buckle": 3, "mini crown": '
                                    '7, "hand on own face": 6, "off-shoulder dress": 3, '
                                    '"shirt lift": 2, "grey panties": 1, "skirt pull": 1, '
                                    '"paint splatter": 3, "lifted by self": 3, "cup": 6, '
                                    '"hammock": 13, "bracelet": 31, "holding spoon": 7, "bead '
                                    'bracelet": 1, "umbrella": 1, "on back": 28, "frilled '
                                    'bikini": 1, "black necktie": 3, "necktie between '
                                    'breasts": 1, "food": 12, "between breasts": 1, "plant": '
                                    '5, "on stomach": 3, "red shorts": 2, "short shorts": 3, '
                                    '"bag": 7, "skirt": 4, "disposable cup": 2, "hand in '
                                    'pocket": 3, "coat": 1, "casual": 4, "bubble tea": 1, '
                                    '"floating hair": 9, "curtains": 3, "white panties": 1, '
                                    '"window": 7, "off-shoulder sweater": 2, "grey sweater": '
                                    '2, "sleeves past wrists": 2, "hair over shoulder": 2, '
                                    '"black sweater": 2, "zipper pull tab": 5, "cat": 1, '
                                    '"food in mouth": 12, "black one-piece swimsuit": 2, '
                                    '"pool": 1, "profile": 24, "blue hair": 1, "2girls": 1, '
                                    '"poolside": 1, "solo focus": 4, "phone screen": 1, '
                                    '"cherry": 30, "hair ornament": 14, "petals": 19, "burnt '
                                    'clothes": 23, "weibo username": 14, "blurry background": '
                                    '4, ":3": 7, "sidelocks": 7, "cropped torso": 7, '
                                    '"lollipop": 4, "sideboob": 23, "chain-link fence": 3, '
                                    '"feet": 11, "no shoes": 5, "sweat": 6, "kneeling": 3, '
                                    '"fang": 3, "front-tie top": 3, "food on body": 1, '
                                    '"front-tie bikini top": 8, "hands up": 11, "stool": 3, '
                                    '"drinking glass": 7, "red footwear": 1, "white gloves": '
                                    '4, "sign": 1, "sky": 18, "orange sky": 2, "sunset": 5, '
                                    '"evening": 1, "butterfly": 2, "butterfly on hand": 1, '
                                    '"holding shoes": 2, "sandals": 6, "flower": 11, "sun '
                                    'hat": 4, "straw hat": 3, "yokozuwari": 1, "headwear '
                                    'removed": 1, "ribbon": 2, "see-through": 10, "holding '
                                    'flower": 2, "halterneck": 12, "red dress": 4, "handbag": '
                                    '2, "cropped jacket": 2, "single hair bun": 3, "red '
                                    'shirt": 2, "covered navel": 7, "holding bottle": 2, '
                                    '"soda bottle": 1, "coca-cola": 2, "vending machine": 2, '
                                    '"tying hair": 6, "hair tie in mouth": 6, "playing '
                                    'instrument": 3, "violin": 3, "holding instrument": 5, '
                                    '"pink flower": 3, "knees up": 14, "arm behind head": 10, '
                                    '"tongue out": 11, "cigarette": 2, "groin": 8, "blood on '
                                    'face": 12, "arms behind head": 6, "red necktie": 2, '
                                    '"collared shirt": 5, "grey shirt": 1, "red rose": 5, '
                                    '"vest": 2, "rose": 5, "red flower": 3, "armlet": 3, '
                                    '"wet": 10, "close-up": 6, "arm strap": 10, "stage": 2, '
                                    '"electric guitar": 2, "concert": 1, "stage lights": 2, '
                                    '"music": 1, "glowstick": 1, "guitar": 1, "from above": '
                                    '7, "looking up": 6, "red panties": 1, "eating": 1, '
                                    '"bra": 3, "underwear only": 4, "champagne flute": 4, '
                                    '"bare back": 9, "small breasts": 1, "black serafuku": 5, '
                                    '"red neckerchief": 5, "crop top overhang": 3, '
                                    '"toenails": 1, "chromatic aberration": 6, "yellow '
                                    'background": 4, "hand on own head": 1, "arm support": 3, '
                                    '"dutch angle": 6, "lipstick tube": 1, "lipstick": 2, '
                                    '"red lips": 5, "purple nails": 2, "lips": 8, "enmaided": '
                                    '3, "white apron": 2, "frills": 2, "black pants": 2, '
                                    '"sneakers": 1, "hip vent": 1, "cleavage cutout": 1, '
                                    '"white bow": 1, "nose blush": 3, "shallow water": 1, '
                                    '"petals on liquid": 1, "white bikini": 9, "sideways '
                                    'glance": 9, "copyright name": 2, "hand on another\'s '
                                    'face": 1, "red choker": 4, "babydoll": 1, "untied '
                                    'panties": 1, "side-tie panties": 1, "white pantyhose": '
                                    '1, "pointy ears": 2, "hand on own stomach": 1, "areola '
                                    'slip": 2, "underbust": 3, "pantyshot": 1, "white pants": '
                                    '2, "blue eyes": 12, "shirt": 2, "grin": 8, "red eyes": '
                                    '12, "playboy bunny": 8, "black bowtie": 3, "rabbit '
                                    'ears": 6, "black leotard": 6, "strapless leotard": 4, '
                                    '"fake animal ears": 6, "thigh gap": 5, "black '
                                    'pantyhose": 10, "brown pantyhose": 2, "fishnet '
                                    'pantyhose": 2, "leotard": 2, "chain necklace": 4, '
                                    '"beach": 12, "on couch": 4, "couch": 8, "halter dress": '
                                    '4, "pelvic curtain": 2, "breasts apart": 3, "plunging '
                                    'neckline": 1, "pocky": 3, ";\\\\)": 2, "tarot": 1, '
                                    '"yellow flower": 2, "bubble": 2, "underwater": 1, "water '
                                    'drop": 1, "hands in hair": 3, "eyewear on head": 17, '
                                    '"red-tinted eyewear": 3, "palm tree": 8, '
                                    '"skindentation": 4, "cameltoe": 2, "sunlight": 6, '
                                    '"string bikini": 1, "orange nails": 1, "hand on own '
                                    'chest": 8, "strapless": 5, "white choker": 6, "elbow '
                                    'gloves": 13, "piercing": 1, "head out of frame": 3, '
                                    '"covered nipples": 4, "looking away": 5, "sundae": 1, '
                                    '"wall clock": 1, "parfait": 1, "fruit": 1, "o-ring": 2, '
                                    '"black bow": 1, "no panties": 1, "bow": 2, "wine glass": '
                                    '9, "chest sarashi": 2, "bandeau": 1, "fishnet gloves": '
                                    '1, "fishnets": 1, "argyle background": 5, "light '
                                    'particles": 1, "holding fruit": 8, "apple": 6, "ass '
                                    'visible through thighs": 3, "shoe soles": 1, "paw '
                                    'print": 1, "bed sheet": 2, "licking": 3, "innertube": 7, '
                                    '"off-shoulder shirt": 3, "ball": 1, "bikini under '
                                    'clothes": 3, "lily pad": 1, "wristband": 1, "two-tone '
                                    'background": 2, "giant": 1, "indian style": 1, "black '
                                    'bra": 4, "electric fan": 1, "head rest": 2, "purple '
                                    'background": 2, "heart": 1, "sunbeam": 5, "light rays": '
                                    '1, "outstretched arms": 1, "crow": 1, "arms under '
                                    'breasts": 1, "breast hold": 2, "pink eyes": 2, "potted '
                                    'plant": 4, "handheld game console": 2, "nintendo '
                                    'switch": 2, "boots removed": 1, "blinds": 1, "single '
                                    'shoe": 1, "dress lift": 1, "sweatdrop": 2, "one eye '
                                    'covered": 2, "rock": 3, "hat": 1, "closed eyes": 6, "red '
                                    'theme": 1, "wrist cuffs": 3, "bar \\\\(place\\\\)": 3, '
                                    '"bottle": 2, "alcohol": 2, "red bowtie": 3, "pink bow": '
                                    '1, "spaghetti strap": 1, "black shorts": 3, "hair in '
                                    'mouth": 1, "zoom layer": 2, "crazy straw": 1, "beach '
                                    'umbrella": 4, "lens flare": 5, "beach chair": 2, "o-ring '
                                    'bikini": 5, "tropical drink": 1, "hand on eyewear": 2, '
                                    '"chair": 1, "adjusting eyewear": 5, "glint": 1, "sand": '
                                    '1, "belt buckle": 2, "ground vehicle": 1, "car": 1, '
                                    '"building": 2, "city": 1, "mountain": 1, "fangs": 3, '
                                    '"teeth": 4, ":d": 3, "chewing gum": 13, "bubble '
                                    'blowing": 12, "black rose": 3, "twintails": 2, "holding '
                                    'phone": 5, "red ribbon": 6, "hairclip": 3, "neck '
                                    'ribbon": 3, "layered dress": 1, "hair flower": 3, "chess '
                                    'piece": 2, "red bow": 2, "king \\\\(chess\\\\)": 1, '
                                    '"white rose": 1, "phone": 1, "rubber duck": 4, "saliva": '
                                    '3, "snot": 1, "belt pouch": 4, "grey dress": 2, "wine": '
                                    '7, "string of fate": 2, "dress pull": 2, "disembodied '
                                    'limb": 1, "clenched teeth": 2, "dress tug": 1, '
                                    '"smartphone": 3, "neckerchief": 1, "cherry blossoms": 3, '
                                    '"maid headdress": 3, "wine bottle": 1, "maid": 2, "legs '
                                    'up": 1, "brown thighhighs": 2, "partially fingerless '
                                    'gloves": 1, "tied shirt": 2, "see-through shirt": 1, '
                                    '"standing on one leg": 1, "horns through headwear": 2, '
                                    '"red bikini": 2, "hand on headwear": 1, "heterochromia": '
                                    '1, "beachball": 1, "sun": 1, "brown headwear": 1, '
                                    '"fence": 1, "frown": 1, "bandaged arm": 1, ":<": 1, '
                                    '"sleeveless shirt": 1, "short hair": 1, "medium hair": '
                                    '1, "purple thighhighs": 1, "tongue": 2, "hand on own '
                                    'thigh": 1, "white thighhighs": 1, "shading eyes": 2, '
                                    '"foot out of frame": 1, "rainbow": 1, "glowing eyes": 2, '
                                    '"garter belt": 2, "lingerie": 2, "star \\\\(sky\\\\)": '
                                    '3, "starry sky": 3, "night": 2, "night sky": 1, "moon": '
                                    '3, "black vest": 2, "weibo logo": 1, "border": 1, '
                                    '"tinted eyewear": 3, ":p": 9, "one knee": 1, "white '
                                    'shorts": 1, "mug": 3, "pajamas": 1, "strap slip": 2, '
                                    '"tiger": 1, "tattoo": 4, "spade \\\\(shape\\\\)": 3, '
                                    '"yellow eyes": 2, "soaking feet": 1, "looking over '
                                    'eyewear": 3, "torn shirt": 1, "upside-down": 1, "holding '
                                    'baseball bat": 1, "baseball bat": 1, "stretching": 3, '
                                    '"wavy mouth": 1, "tears": 1, "highleg leotard": 2, '
                                    '"purple hair": 2, "puffy sleeves": 1, "no pants": 1, '
                                    '"halo": 1, "rhodes island logo": 1, "bokeh": 2, "center '
                                    'opening": 1}}',
                'ss_text_encoder_lr': '0.0006',
                'ss_total_batch_size': '8',
                'ss_training_comment': 'nebulae',
                'ss_training_finished_at': '1710330868.748919',
                'ss_training_started_at': '1710328450.662775',
                'ss_unet_lr': '0.0006',
                'ss_v2': 'False',
                'ss_zero_terminal_snr': 'False',
                'sshs_legacy_hash': '147fd55d',
                'sshs_model_hash': 'bullshit',
            }
