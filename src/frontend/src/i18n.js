import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  vi: {
    translation: {
      dashboard: 'Tổng quan',
      patients: 'Bệnh nhân',
      dosing: 'Tính liều',
      results: 'Kết quả',
      validation: 'Thẩm định',
      settings: 'Cài đặt'
    }
  }
}

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: 'vi',
    fallbackLng: 'vi',
    interpolation: { escapeValue: false }
  })

export default i18n
