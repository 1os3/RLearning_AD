# 创建__init__.py文件
@(
    "src\models\fusion_module\__init__.py",
    "src\models\policy_module\__init__.py",
    "src\environment\__init__.py",
    "src\algorithms\__init__.py",
    "src\utils\__init__.py",
    "tests\__init__.py"
) | ForEach-Object {
    New-Item -Path $_ -ItemType File -Force
    Write-Host "Created $_"
}

# 创建配置文件
@(
    "config\default.yaml",
    "config\training.yaml",
    "config\inference.yaml"
) | ForEach-Object {
    New-Item -Path $_ -ItemType File -Force
    Write-Host "Created $_"
}

# 创建README文件
New-Item -Path "README.md" -ItemType File -Force
Write-Host "Created README.md"

# 创建核心Python文件
@(
    # Input Module
    "src\models\input_module\frame_processor.py",
    "src\models\input_module\pose_encoder.py",
    "src\models\input_module\target_encoder.py",
    
    # Fusion Module
    "src\models\fusion_module\depthwise_conv.py",
    "src\models\fusion_module\cross_attention.py",
    "src\models\fusion_module\memory_bank.py",
    
    # Policy Module
    "src\models\policy_module\actor.py",
    "src\models\policy_module\critic.py",
    
    # Environment
    "src\environment\airsim_env.py",
    "src\environment\reward.py",
    
    # Algorithms
    "src\algorithms\sac.py",
    "src\algorithms\replay_buffer.py",
    "src\algorithms\auxiliary.py",
    
    # Utils
    "src\utils\checkpoint.py",
    "src\utils\visualization.py",
    "src\utils\domain_random.py",
    "src\utils\metrics.py",
    
    # Scripts
    "scripts\train.py",
    "scripts\inference.py",
    "scripts\evaluate.py",
    
    # Tests
    "tests\test_models.py",
    "tests\test_environment.py",
    "tests\test_algorithms.py"
) | ForEach-Object {
    New-Item -Path $_ -ItemType File -Force
    Write-Host "Created $_"
}

Write-Host "Project structure created successfully!"
