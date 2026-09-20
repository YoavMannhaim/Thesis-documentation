# Experiments — Photo and Video Documentation

Raw photo and video documentation from the physical experiments behind this
thesis. This is supporting media, not data — the actual measurements and
analysis are in the thesis document and, where applicable, alongside the
code in `Grasp_Validation/`.

```
Experiments/
├── Material_Characterization/
│   ├── Thickness_Measurement/     11 photos — micrometer readings, thesis §5.3.1
│   ├── Simple_Tensile_Test/       17 photos — Young's modulus rig, thesis §5.3.2
│   ├── Simple_Shear_Test/         22 photos — shear modulus rig, thesis §5.3.3
│   └── Surface_Energy_Test/       19 photos, 3 videos — peel test, thesis §5.3.4
├── Sensor_Selection/               5 photos, 31 videos — IR vs. photoresistor
│                                   comparison, thesis §7.1
└── Grasp_Validation/               5 photos, 3 videos — hardware pilot on the
                                    KUKA rig, thesis §7.4.5
```

Each subfolder's files are numbered in the order they were originally
recorded (`photo_01`, `photo_02`, ... / `video_01`, `video_02`, ...).

**Grasp_Validation is duplicated on purpose.** The same 5 photos and 3
videos here also live in `Grasp_Validation/hardware_validation/final/photos/`
and `.../videos/`, alongside the sensor recordings and trained models those
sessions produced — so that folder is a complete, self-contained record of
the hardware pilot without needing to look elsewhere. This folder keeps the
copy that sits alongside every other experiment for anyone browsing the
physical documentation as a whole.
