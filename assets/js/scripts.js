/**
 * Site-wide JS
 */

// Run MathJax when Turbolinks navigates to a page
document.addEventListener('turbolinks:load', () => {
  if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub])
})

const togglerId = 'js-sidebar-toggle'
const textbookId = 'js-textbook'
const togglerActiveClass = 'is-active'
const textbookActiveClass = 'js-show-sidebar'

const getToggler = () => document.getElementById(togglerId)
const getTextbook = () => document.getElementById(textbookId)

/**
 * Toggles sidebar and menu icon
 */
const toggleSidebar = () => {
  const toggler = getToggler()
  const textbook = getTextbook()

  if (textbook.classList.contains(textbookActiveClass)) {
    textbook.classList.remove(textbookActiveClass)
    toggler.classList.remove(togglerActiveClass)
  } else {
    textbook.classList.add(textbookActiveClass)
    toggler.classList.add(togglerActiveClass)
  }
}

/**
 * Auto-close sidebar on smaller screens after page load.
 *
 * Having the sidebar be open by default then closing it on page load for
 * small screens gives the illusion that the sidebar closes in response
 * to selecting a page in the sidebar. However, it does cause a bit of jank
 * on the first page load.
 *
 * Since we don't want to persist state in between page navigation, this is
 * the best we can do while optimizing for larger screens where most
 * viewers will read the textbook.
 *
 * Keep the variable below in sync with the tablet breakpoint value in
 * _sass/inuitcss/tools/_tools.mq.scss
 *
 */
const autoCloseSidebarBreakpoint = 740

// Set up event listener for sidebar toggle button
document.addEventListener('turbolinks:load', () => {
  getToggler().addEventListener('click', toggleSidebar)

  // This assumes that the sidebar is open by default
  if (window.innerWidth < autoCloseSidebarBreakpoint) toggleSidebar()
})

/**
 * Preserve sidebar scroll when navigating between pages
 */
let sidebarScrollTop = 0
const getSidebar = () => document.getElementById('js-sidebar')

document.addEventListener('turbolinks:before-visit', () => {
  sidebarScrollTop = getSidebar().scrollTop
})

document.addEventListener('turbolinks:load', () => {
  getSidebar().scrollTop = sidebarScrollTop
})

/**
 * Focus textbook page by default so that user can scroll with spacebar
 */
document.addEventListener('turbolinks:load', () => {
  document.querySelector('.c-textbook__page').focus()
})

/**
 * Use left and right arrow keys to navigate forward and backwards.
 */
const LEFT_ARROW_KEYCODE = 37
const RIGHT_ARROW_KEYCODE = 39

const getPrevUrl = () => document.getElementById('js-page__nav__prev').href
const getNextUrl = () => document.getElementById('js-page__nav__next').href
document.addEventListener('keydown', event => {
  const keycode = event.which

  if (keycode === LEFT_ARROW_KEYCODE) {
    Turbolinks.visit(getPrevUrl())
  } else if (keycode === RIGHT_ARROW_KEYCODE) {
    Turbolinks.visit(getNextUrl())
  }
})
