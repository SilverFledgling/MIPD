-- ══════════════════════════════════════════════════════════════════
-- MIPD Database Schema — MS SQL Server
-- Chuẩn BCNF + 4NF
--
-- BCNF: Mọi phụ thuộc hàm X → Y đều có X là siêu khóa
-- 4NF:  Không có phụ thuộc đa trị (tách multi-valued → bảng riêng)
--
-- Run:  sqlcmd -S localhost -d master -i schema.sql
-- ══════════════════════════════════════════════════════════════════

-- ── Tạo Database ───────────────────────────────────────────────
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'MIPD')
    CREATE DATABASE MIPD;
GO

USE MIPD;
GO

-- ────────────────────────────────────────────────────────────────
-- 1. Users (Tài khoản người dùng)
--    BCNF: UserId → all columns; Email UNIQUE (candidate key)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE Users (
    UserId       UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    Email        NVARCHAR(200)    NOT NULL UNIQUE,
    PasswordHash NVARCHAR(500)    NOT NULL,
    FullName     NVARCHAR(200)    NOT NULL,
    Role         NVARCHAR(20)     NOT NULL
                 CHECK (Role IN ('Admin','Physician','Pharmacist','Nurse')),
    IsActive     BIT              NOT NULL DEFAULT 1,
    CreatedAt    DATETIME2        NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt    DATETIME2        NULL
);
GO

-- ────────────────────────────────────────────────────────────────
-- 2. Patients (Thông tin nhân khẩu học — bất biến hoặc hiếm đổi)
--    BCNF: PatientId → all; MRN UNIQUE (candidate key)
--    4NF:  Weight/Height tách sang ClinicalObservations (thay đổi theo thời gian)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE Patients (
    PatientId   UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    MRN         NVARCHAR(50)     NOT NULL UNIQUE,  -- Medical Record Number
    FullName    NVARCHAR(200)    NOT NULL,
    DateOfBirth DATE             NOT NULL,
    Gender      TINYINT          NOT NULL CHECK (Gender IN (0, 1, 2)),
                                 -- 0=Unknown, 1=Male, 2=Female (HL7)
    CreatedAt   DATETIME2        NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt   DATETIME2        NULL,
    CreatedBy   UNIQUEIDENTIFIER NULL REFERENCES Users(UserId)
);
GO

-- ────────────────────────────────────────────────────────────────
-- 3. ClinicalObservations (Xét nghiệm + chỉ số lâm sàng)
--    BCNF: ObservationId → all
--    4NF:  Mỗi dòng = 1 chỉ số lâm sàng tại 1 thời điểm
--          (tách weight, creatinine, albumin... thành từng dòng)
--    → Thiết kế EAV (Entity-Attribute-Value) cho linh hoạt
-- ────────────────────────────────────────────────────────────────
CREATE TABLE ClinicalObservations (
    ObservationId UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId     UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    ObsType       NVARCHAR(50)     NOT NULL,
                  -- Nhân khẩu học: 'Weight','Height','BMI'
                  -- Chức năng thận: 'SerumCreatinine','CrCL','eGFR'
                  -- Chức năng gan: 'ALT','AST','TotalBilirubin','Albumin'
                  -- Huyết động (CV 1.1): 'HeartRate','MAP_mmHg','SystolicBP','DiastolicBP'
                  -- Nhiễm khuẩn (CV 1.1): 'WBC','CRP','Procalcitonin','SOFA_Score'
    ObsValue      DECIMAL(10,4)    NOT NULL,
    ObsUnit       NVARCHAR(20)     NOT NULL,
                  -- 'kg','cm','mg/dL','mL/min','g/dL','U/L'
    RecordedAt    DATETIME2        NOT NULL,
    CreatedAt     DATETIME2        NOT NULL DEFAULT GETUTCDATE()
);
GO

CREATE INDEX IX_ClinObs_Patient_Type
    ON ClinicalObservations(PatientId, ObsType, RecordedAt DESC);
GO

-- ────────────────────────────────────────────────────────────────
-- 4. Comorbidities (Bệnh đi kèm — 4NF: tách multi-valued)
--    Trước: JSON array trong ClinicalData → vi phạm 4NF
--    Sau:   Mỗi dòng = 1 bệnh đi kèm
-- ────────────────────────────────────────────────────────────────
CREATE TABLE PatientComorbidities (
    Id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId   UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    ICD10Code   NVARCHAR(10)     NULL,       -- Mã ICD-10
    Description NVARCHAR(200)    NOT NULL,    -- Tên bệnh
    IsActive    BIT              NOT NULL DEFAULT 1,
    RecordedAt  DATETIME2        NOT NULL DEFAULT GETUTCDATE()
);
GO

CREATE INDEX IX_Comorbidity_Patient
    ON PatientComorbidities(PatientId);
GO

-- ────────────────────────────────────────────────────────────────
-- 5. ConcomitantDrugs (Thuốc dùng kèm — 4NF: tách multi-valued)
--    Trước: JSON array → vi phạm 4NF
--    Sau:   Mỗi dòng = 1 thuốc dùng kèm
-- ────────────────────────────────────────────────────────────────
CREATE TABLE PatientConcomitantDrugs (
    Id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId   UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    DrugName    NVARCHAR(200)    NOT NULL,
    DoseInfo    NVARCHAR(100)    NULL,  -- VD: "500mg BID"
    StartDate   DATE             NULL,
    EndDate     DATE             NULL,
    RecordedAt  DATETIME2        NOT NULL DEFAULT GETUTCDATE()
);
GO

CREATE INDEX IX_ConcomDrug_Patient
    ON PatientConcomitantDrugs(PatientId);
GO

-- ────────────────────────────────────────────────────────────────
-- 6. DrugAdministrations (Lịch sử dùng thuốc TDM)
--    BCNF: Id → all
-- ────────────────────────────────────────────────────────────────
CREATE TABLE DrugAdministrations (
    Id               UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId        UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    DrugName         NVARCHAR(100)    NOT NULL,  -- 'vancomycin'
    Dose             DECIMAL(8,2)     NOT NULL,  -- mg
    InfusionDuration DECIMAL(6,2)     NULL,      -- hours (NULL = bolus)
    Route            NVARCHAR(20)     NOT NULL DEFAULT 'IV',
    AdministeredAt   DATETIME2        NOT NULL,
    CreatedAt        DATETIME2        NOT NULL DEFAULT GETUTCDATE()
);
GO

CREATE INDEX IX_DrugAdmin_Patient_Drug
    ON DrugAdministrations(PatientId, DrugName, AdministeredAt DESC);
GO

-- ────────────────────────────────────────────────────────────────
-- 7. TDMSamples (Mẫu nồng độ thuốc — TDM)
--    BCNF: Id → all
-- ────────────────────────────────────────────────────────────────
CREATE TABLE TDMSamples (
    Id            UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId     UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    DrugName      NVARCHAR(100)    NOT NULL,
    Concentration DECIMAL(8,3)     NOT NULL,  -- mg/L (μg/mL)
    SampledAt     DATETIME2        NOT NULL,
    SampleType    NVARCHAR(20)     NOT NULL
                  CHECK (SampleType IN ('trough','peak','midpoint','random')),
    CreatedAt     DATETIME2        NOT NULL DEFAULT GETUTCDATE()
);
GO

CREATE INDEX IX_TDM_Patient_Drug
    ON TDMSamples(PatientId, DrugName, SampledAt DESC);
GO

-- ────────────────────────────────────────────────────────────────
-- 8. DosingRecommendations (Khuyến nghị liều)
--    BCNF: Id → all
--    4NF:  ConfidenceInterval tách thành 2 cột scalar (lower, upper)
--          ShapValues tách sang bảng riêng
-- ────────────────────────────────────────────────────────────────
CREATE TABLE DosingRecommendations (
    Id               UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId        UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    DrugName         NVARCHAR(100)    NOT NULL,
    BayesianMethod   NVARCHAR(50)     NOT NULL,
                     -- 'MAP','MCMC_NUTS','Laplace','ADVI','SMC','BMA','HB'
    RecommendedDose  DECIMAL(8,2)     NOT NULL,  -- mg
    InfusionDuration DECIMAL(6,2)     NULL,      -- hours
    DosingInterval   DECIMAL(6,2)     NOT NULL,  -- hours
    PredictedAUC24   DECIMAL(10,2)    NULL,
    PredictedTrough  DECIMAL(8,3)     NULL,      -- mg/L
    TargetAUCMIC     DECIMAL(8,2)     NULL,
    EstimatedCL      DECIMAL(8,4)     NULL,      -- L/h
    EstimatedVd      DECIMAL(8,4)     NULL,      -- L
    CI_Lower95       DECIMAL(8,3)     NULL,      -- 95% CI lower bound
    CI_Upper95       DECIMAL(8,3)     NULL,      -- 95% CI upper bound
    CreatedAt        DATETIME2        NOT NULL DEFAULT GETUTCDATE(),
    CreatedBy        UNIQUEIDENTIFIER NULL REFERENCES Users(UserId)
);
GO

CREATE INDEX IX_DosingRec_Patient_Drug
    ON DosingRecommendations(PatientId, DrugName, CreatedAt DESC);
GO

-- ────────────────────────────────────────────────────────────────
-- 9. ShapValues (Giải thích SHAP — 4NF: tách multi-valued)
--    Trước: JSON blob trong DosingRecommendations → vi phạm 4NF
--    Sau:   Mỗi dòng = 1 SHAP value cho 1 feature
-- ────────────────────────────────────────────────────────────────
CREATE TABLE ShapValues (
    Id                 UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    RecommendationId   UNIQUEIDENTIFIER NOT NULL
                       REFERENCES DosingRecommendations(Id),
    FeatureName        NVARCHAR(100)    NOT NULL,  -- 'Weight','CrCL','Age',...
    FeatureValue       DECIMAL(10,4)    NULL,
    ShapValue          DECIMAL(10,6)    NOT NULL,  -- Contribution to prediction
    Rank               INT              NULL       -- Importance rank
);
GO

CREATE INDEX IX_Shap_Recommendation
    ON ShapValues(RecommendationId);
GO

-- ────────────────────────────────────────────────────────────────
-- 10. AuditLog (Nhật ký thao tác)
--     BCNF: Id → all
-- ────────────────────────────────────────────────────────────────
CREATE TABLE AuditLog (
    Id         BIGINT IDENTITY(1,1) PRIMARY KEY,
    UserId     UNIQUEIDENTIFIER NULL REFERENCES Users(UserId),
    Action     NVARCHAR(100)    NOT NULL,  -- 'CREATE','READ','UPDATE','DELETE'
    EntityType NVARCHAR(100)    NOT NULL,  -- 'Patient','DosingRecommendation',...
    EntityId   NVARCHAR(100)    NULL,
    Details    NVARCHAR(MAX)    NULL,
    IpAddress  NVARCHAR(50)     NULL,
    Timestamp  DATETIME2        NOT NULL DEFAULT GETUTCDATE()
);
GO

CREATE INDEX IX_Audit_User_Time
    ON AuditLog(UserId, Timestamp DESC);
CREATE INDEX IX_Audit_Entity
    ON AuditLog(EntityType, EntityId);
GO

-- ── Bảng tham chiếu: InfectionSites ────────────────────────────
-- BCNF: SiteId → SiteName (1:1 candidate key)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE InfectionSites (
    SiteId   INT IDENTITY(1,1) PRIMARY KEY,
    SiteName NVARCHAR(100)     NOT NULL UNIQUE
);
GO

-- Dữ liệu mẫu
INSERT INTO InfectionSites (SiteName) VALUES
    (N'Phổi'),
    (N'Máu (Nhiễm khuẩn huyết)'),
    (N'Da và mô mềm'),
    (N'Xương khớp'),
    (N'Nội tâm mạc'),
    (N'Hệ thần kinh trung ương'),
    (N'Đường tiết niệu'),
    (N'Ổ bụng'),
    (N'Khác');
GO

-- Bảng liên kết Patient ↔ InfectionSite (cho phép nhiều vị trí)
CREATE TABLE PatientInfectionSites (
    Id           UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    PatientId    UNIQUEIDENTIFIER NOT NULL REFERENCES Patients(PatientId),
    SiteId       INT              NOT NULL REFERENCES InfectionSites(SiteId),
    DiagnosedAt  DATETIME2        NOT NULL DEFAULT GETUTCDATE(),
    IsActive     BIT              NOT NULL DEFAULT 1,
    CONSTRAINT UQ_Patient_Site UNIQUE (PatientId, SiteId, DiagnosedAt)
);
GO

PRINT '══════════════════════════════════════════════════════════';
PRINT '  MIPD Database schema created successfully!';
PRINT '  Tables: 11 (BCNF + 4NF compliant)';
PRINT '══════════════════════════════════════════════════════════';
GO
